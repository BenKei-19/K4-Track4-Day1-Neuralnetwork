"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, activation_stats, count_params
from optimizer import build_optimizer, build_scheduler, grad_norm_tensor

N_CLASSES = 7
TRAIN_EVAL_SIZE = 50_000   # train loss đo trên 50 000 mẫu ĐẦU của X_tr: cố định cho mọi epoch và mọi thí nghiệm

# Cấu hình mặc định = BASELINE (M-base). `lr` được chọn bằng val trong notebook (Part 2) rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    scheduler=None,            # None | "cosine" (chỉ dùng ở cấu hình cuối cùng)
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def per_class_scores(cm: np.ndarray):
    """precision, recall, F1 của từng lớp từ ma trận nhầm lẫn (hàng = thật, cột = dự đoán), như evaluate.py."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    return float(per_class_scores(cm)[2].mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits (chế độ eval, không dropout)."""
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)])


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1, cm) ở chế độ eval() (dropout tắt), FP32 và no_grad.

    Loss được cộng dồn theo tổng rồi chia N (không lấy trung bình của trung bình các lô).
    Dùng hàm này cho: train loss (tập con cố định của train), val, và eval cuối cùng.
    """
    model.eval()
    loss_sum = torch.zeros((), device=X.device, dtype=torch.float64)
    cm = torch.zeros(N_CLASSES * N_CLASSES, device=X.device, dtype=torch.int64)
    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb)
        loss_sum += compute_loss(logits, yb, loss_name).double() * len(yb)
        pred = logits.argmax(dim=1)
        cm += torch.bincount(yb * N_CLASSES + pred, minlength=N_CLASSES * N_CLASSES)
    cm = cm.view(N_CLASSES, N_CLASSES).cpu().numpy()
    return dict(loss=float(loss_sum.item() / len(X)),
                acc=float(np.trace(cm) / cm.sum()),
                macro_f1=macro_f1_from_confusion(cm),
                cm=cm)


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy, softmax nằm bên trong).
       "mse" : MSE giữa logit và one-hot của y, giống nn.MSELoss mặc định: trung bình trên MỌI phần tử
               (B x 7), không có hệ số 1/2. Dự đoán vẫn là argmax của logit.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        return F.mse_loss(logits, F.one_hot(y, N_CLASSES).to(logits.dtype))
    raise ValueError(f"loss không hợp lệ: {loss_name!r}")


def _nan_if_none(x):
    return float("nan") if x is None else x


def run_experiment(cfg: dict, data: dict, verbose: bool = False) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (các khoá thiếu lấy từ DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                     "grad_norm" (TB mỗi epoch, đo TRƯỚC khi clip), "grad_norm_max", "clip_frac", "epoch_time_s"},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged", ... (thêm vài khoá phụ)},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Quy ước đo:
      - train_loss đo ở chế độ eval() trên TRAIN_EVAL_SIZE mẫu đầu của X_tr (tập con cố định), cùng thang với val.
      - grad_norm: chuẩn L2 toàn cục, đo TRƯỚC khi clip (với fp16: sau unscale_). Các bước fp16 bị GradScaler
        bỏ qua (grad inf/NaN) không được tính vào trung bình.
      - epoch_time_s: chỉ tính thời gian huấn luyện của epoch (có torch.cuda.synchronize()), không tính phần đánh giá.
      - peak_mem_MB: bộ nhớ GPU cực đại TRỪ phần đã cấp phát trước khi tạo model (tức là không tính dữ liệu).
      - Mọi số ở summary lấy tại best_epoch (val_loss thấp nhất) — giống dừng sớm.
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    if cfg["lr"] is None:
        raise ValueError("cfg['lr'] chưa được đặt")
    hidden = tuple(cfg["hidden"])
    cfg["hidden"] = list(hidden)
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    X_sub, y_sub = X_tr[:TRAIN_EVAL_SIZE], y_tr[:TRAIN_EVAL_SIZE]
    device = X_tr.device
    use_cuda = device.type == "cuda"

    # ---- 0. model, optimizer, scaler (mọi thứ sinh ngẫu nhiên đều theo cfg["seed"])
    set_seed(cfg["seed"])
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device)
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"]).to(device)
    n_params = count_params(model)
    if hidden in EXPECTED_PARAMS:
        assert n_params == EXPECTED_PARAMS[hidden], (n_params, EXPECTED_PARAMS[hidden])
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], total_steps=steps_per_epoch * cfg["epochs"])
    amp_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[cfg["precision"]]
    # GradScaler chỉ bật cho fp16; khi tắt thì scale/unscale_/step/update trở thành optimizer.step() bình thường
    scaler = torch.amp.GradScaler(device.type, enabled=(cfg["precision"] == "fp16"))
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg["seed"])

    def autocast():
        if amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    # ---- 1. bước 0: TRƯỚC bước cập nhật đầu tiên
    step0 = evaluate(model, X_val, y_val, cfg["loss"])
    act_std0 = activation_stats(model, X_val[:4096])

    history = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                               "grad_norm", "grad_norm_max", "clip_frac", "epoch_time_s")}
    best_val_loss, best_epoch, best_state = math.inf, None, None
    diverged = False
    all_gn = []

    # ---- 2. huấn luyện
    for epoch in range(1, cfg["epochs"] + 1):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train()
        gn_steps = []                                              # tensor 0 chiều trên GPU, đọc 1 lần/epoch
        bad_loss = torch.zeros((), dtype=torch.bool, device=device)
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
            with autocast():                                       # chỉ bọc forward + loss
                loss = compute_loss(model(xb), yb, cfg["loss"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                             # fp16: đưa grad về thang thật TRƯỚC khi đo/cắt
            gn_steps.append(grad_norm_tensor(model.parameters(), cfg["clip_norm"]))   # chuẩn TRƯỚC khi cắt
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            bad_loss |= ~torch.isfinite(loss.detach())
        if use_cuda:
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0

        gn = torch.stack(gn_steps).float()
        gn = gn[torch.isfinite(gn)]
        all_gn.append(gn)
        clip_frac = float((gn > cfg["clip_norm"]).float().mean()) if cfg["clip_norm"] is not None and len(gn) else 0.0

        tr = evaluate(model, X_sub, y_sub, cfg["loss"])
        va = evaluate(model, X_val, y_val, cfg["loss"])
        history["epoch"].append(epoch)
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        history["val_macro_f1"].append(va["macro_f1"])
        history["grad_norm"].append(float(gn.mean()) if len(gn) else float("nan"))
        history["grad_norm_max"].append(float(gn.max()) if len(gn) else float("nan"))
        history["clip_frac"].append(clip_frac)
        history["epoch_time_s"].append(epoch_time)
        if verbose:
            print(f"  epoch {epoch:2d}  train {tr['loss']:.4f}  val {va['loss']:.4f}  "
                  f"acc {va['acc']:.4f}  f1 {va['macro_f1']:.4f}  gn {history['grad_norm'][-1]:.3f}  {epoch_time:.1f}s")

        if bool(bad_loss) or not math.isfinite(va["loss"]):
            diverged = True                                        # dừng sớm, đừng để notebook treo
            break
        if va["loss"] < best_val_loss:
            best_val_loss, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # ---- 3. tóm tắt tại best_epoch
    gn_all = torch.cat(all_gn) if all_gn else torch.zeros(0)
    i = best_epoch - 1 if best_epoch is not None else None
    summary = dict(
        step0_loss=step0["loss"],
        best_val_loss=best_val_loss if best_epoch is not None else float("nan"),
        best_epoch=best_epoch,
        final_train_loss=history["train_loss"][-1],
        final_val_loss=history["val_loss"][-1],
        val_acc=history["val_acc"][i] if i is not None else float("nan"),
        val_macro_f1=history["val_macro_f1"][i] if i is not None else float("nan"),
        time_per_epoch_s=float(np.mean(history["epoch_time_s"])),
        peak_mem_MB=(torch.cuda.max_memory_allocated(device) - mem_before) / 2**20 if use_cuda else float("nan"),
        diverged=diverged,
        # khoá phụ (không phải cột bắt buộc; dùng cho notes và phân tích)
        n_params=n_params,
        steps_per_epoch=steps_per_epoch,
        step0_acc=step0["acc"],
        act_std_step0=act_std0,
        grad_norm_p50=float(gn_all.median()) if len(gn_all) else float("nan"),
        grad_norm_p90=float(torch.quantile(gn_all, 0.9)) if len(gn_all) else float("nan"),
        clip_frac_mean=float(np.mean(history["clip_frac"])),
        fp16_skipped_steps=int(cfg["epochs"] * steps_per_epoch - len(gn_all)) if cfg["precision"] == "fp16" else 0,
    )
    s = summary
    print(f"{cfg['exp_id']:22s} val_f1={_nan_if_none(s['val_macro_f1']):.4f} val_acc={_nan_if_none(s['val_acc']):.4f} "
          f"best_ep={s['best_epoch']} step0={s['step0_loss']:.3f} t/ep={s['time_per_epoch_s']:.2f}s"
          + ("  ** DIVERGED **" if diverged else ""))
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_id = np.asarray(row_id, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    assert row_id.shape == preds.shape and row_id.ndim == 1
    assert len(np.unique(row_id)) == len(row_id), "row_id bị lặp"
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1, "pred phải nằm trong 0..6"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savetxt(path, np.column_stack([row_id, preds]), fmt="%d", delimiter=",", header="row_id,pred", comments="")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str,
               repo_root: str | None = None, out_json: str | None = None) -> dict | None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. nếu có repo_root và out_json: chạy `python scripts/evaluate.py --pred <pred_path> --out <out_json>`
         (đúng script giảng viên dùng để chấm), in kết quả và trả về dict đọc từ out_json
    """
    assert result.get("best_state") is not None, "cần best_state (chạy lại run_experiment trong phiên này)"
    device = data["X_eval"].device
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=cfg["dropout"], init="default").to(device)
    model.load_state_dict(result["best_state"])
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    if repo_root is None or out_json is None:
        return None
    proc = subprocess.run([sys.executable, "scripts/evaluate.py", "--pred", os.path.abspath(pred_path),
                           "--out", os.path.abspath(out_json)],
                          cwd=repo_root, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"evaluate.py lỗi:\n{proc.stdout}\n{proc.stderr}")
    with open(out_json, encoding="utf-8") as f:
        return json.load(f)
