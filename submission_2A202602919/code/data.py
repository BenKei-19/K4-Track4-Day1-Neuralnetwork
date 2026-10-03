"""data.py — nạp train/eval đã chia sẵn, tách validation, chuẩn hoá, đưa lên thiết bị, cắt lô.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations
import numpy as np
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    train = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")
    X_train_full, y_train_full = train["X"], train["y"]
    X_eval, y_eval, eval_row_id = ev["X"], ev["y"], ev["row_id"]
    for X, y in [(X_train_full, y_train_full), (X_eval, y_eval)]:
        assert X.dtype == np.float32 and X.shape == (len(y), N_NUMERIC + 44)
        assert y.dtype == np.int64 and y.ndim == 1
        assert y.min() == 0 and y.max() == 6
    assert len(eval_row_id) == len(y_eval)
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_fraction, random_state=seed, stratify=y)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    mean = np.mean(X_tr[:, :N_NUMERIC], axis=0)
    std = np.std(X_tr[:, :N_NUMERIC], axis=0)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X_out = X.copy()
    X_out[:, :N_NUMERIC] = (X_out[:, :N_NUMERIC] - mean) / (std + 1e-8)
    return X_out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    from sklearn.metrics import f1_score

    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction, seed)

    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    majority = int(np.bincount(y_tr).argmax())
    maj_acc = float((y_val == majority).mean())
    maj_f1 = f1_score(y_val, np.full_like(y_val, majority), average="macro", zero_division=0)
    print(f"train: {len(y_tr)} | val: {len(y_val)} | eval: {len(y_eval)}")
    print(f"Luôn đoán lớp {majority}: val acc = {maj_acc:.4f}, val macro-F1 = {maj_f1:.4f}")

    def to_x(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    def to_y(a):
        return torch.tensor(a, dtype=torch.int64, device=device)

    return {
        "X_tr": to_x(X_tr), "y_tr": to_y(y_tr),
        "X_val": to_x(X_val), "y_val": to_y(y_val),
        "X_eval": to_x(X_eval), "y_eval": to_y(y_eval),
        "eval_row_id": eval_row_id,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    N = len(X)
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=X.device)
    else:
        perm = torch.arange(N, device=X.device)
    # Giữ lô cuối dù nhỏ hơn batch_size (N=371 847, batch=512 -> lô cuối 135 mẫu) để mỗi epoch dùng hết dữ liệu
    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        yield X[idx], y[idx]
    
