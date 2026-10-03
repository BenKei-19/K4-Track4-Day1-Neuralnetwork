"""plots.py — ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.

Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

MAJORITY_ACC = 0.4876   # mốc "luôn đoán lớp đa số" trên val

LABELS = {
    "train_loss": "train loss (eval mode)",
    "val_loss": "val loss",
    "val_acc": "val accuracy",
    "val_macro_f1": "val macro-F1",
    "grad_norm": "grad norm TB/epoch (trước clip)",
    "grad_norm_max": "grad norm lớn nhất/epoch (trước clip)",
    "clip_frac": "tỉ lệ bước bị cắt",
    "epoch_time_s": "thời gian/epoch (s)",
    "gap": "val loss − train loss",
}
OPT_NAMES = {"sgd": "SGD", "sgd_momentum": "SGD+mom", "adam": "Adam", "adamw": "AdamW"}


def cfg_text(cfg: dict) -> str:
    """Mô tả ngắn cấu hình để đặt vào tiêu đề ảnh."""
    parts = [f"{OPT_NAMES.get(cfg['optimizer'], cfg['optimizer'])} lr={cfg['lr']:g}",
             f"loss={cfg['loss'].upper()}", f"batch={cfg['batch']}",
             f"hidden={'-'.join(str(h) for h in cfg['hidden'])}", f"init={cfg['init']}"]
    if cfg.get("weight_decay"):
        parts.append(f"wd={cfg['weight_decay']:g}")
    if cfg.get("dropout"):
        parts.append(f"dropout q={cfg['dropout']:g}")
    if cfg.get("clip_norm") is not None:
        parts.append(f"clip c={cfg['clip_norm']:g}")
    if cfg.get("precision", "fp32") != "fp32":
        parts.append(cfg["precision"])
    if cfg.get("scheduler"):
        parts.append(f"lr {cfg['scheduler']}")
    parts += [f"{cfg['epochs']} epoch", f"seed={cfg['seed']}"]
    return ", ".join(parts)


def _fmt(v, nd: int = 4) -> str:
    """Định dạng số cho tiêu đề; None/NaN (lần chạy phân kỳ) -> "—"."""
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _series(result: dict, metric: str) -> np.ndarray:
    h = result["history"]
    if metric == "gap":
        return np.array(h["val_loss"], dtype=float) - np.array(h["train_loss"], dtype=float)
    return np.array(h[metric], dtype=float)          # None (NaN đã lưu JSON) -> nan


def _auto_log(ax, values) -> None:
    v = np.concatenate([np.ravel(x) for x in values]) if values else np.array([])
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) and v.max() / v.min() > 30:
        ax.set_yscale("log")


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc và val_macro_f1 theo epoch (kèm mốc "đoán lớp đa số")
         (3) grad_norm (trung bình và lớn nhất mỗi epoch, đo TRƯỚC khi clip) và ngưỡng c nếu có clip
    Tiêu đề ghi exp_id và cấu hình chính; đường thẳng đứng đánh dấu best_epoch.
    """
    cfg, s = result["cfg"], result["summary"]
    ep = np.array(result["history"]["epoch"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    ax = axes[0]
    tr, va = _series(result, "train_loss"), _series(result, "val_loss")
    ax.plot(ep, tr, "o-", ms=3, label=LABELS["train_loss"])
    ax.plot(ep, va, "o-", ms=3, label=LABELS["val_loss"])
    _auto_log(ax, [tr, va])
    ax.set(xlabel="epoch", ylabel=f"loss ({cfg['loss'].upper()})",
           title=f"Loss (bước 0 trên val = {_fmt(s['step0_loss'], 3)})")

    ax = axes[1]
    ax.plot(ep, _series(result, "val_acc"), "o-", ms=3, label=LABELS["val_acc"])
    ax.plot(ep, _series(result, "val_macro_f1"), "o-", ms=3, label=LABELS["val_macro_f1"])
    ax.axhline(MAJORITY_ACC, ls=":", c="gray", label="đoán lớp đa số (acc 0.4876)")
    ax.set(xlabel="epoch", ylabel="giá trị (0–1)", title="Val accuracy và macro-F1")

    ax = axes[2]
    gn, gmax = _series(result, "grad_norm"), _series(result, "grad_norm_max")
    ax.plot(ep, gn, "o-", ms=3, label=LABELS["grad_norm"])
    ax.plot(ep, gmax, "^--", ms=3, alpha=0.7, label=LABELS["grad_norm_max"])
    if cfg.get("clip_norm") is not None:
        ax.axhline(cfg["clip_norm"], ls="--", c="red",
                   label=f"ngưỡng clip c = {cfg['clip_norm']:g} (cắt {100 * s.get('clip_frac_mean', 0):.0f}% số bước)")
    _auto_log(ax, [gn, gmax])
    ax.set(xlabel="epoch", ylabel="chuẩn L2 toàn cục của gradient", title="Grad norm (đo trước khi clip)")

    for ax in axes:
        if s.get("best_epoch"):
            ax.axvline(s["best_epoch"], ls="--", c="green", alpha=0.5, label=f"best epoch = {s['best_epoch']}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    status = "  [DIVERGED]" if s.get("diverged") else ""
    fig.suptitle(f"{cfg['exp_id']}{status} — {cfg_text(cfg)}\n"
                 f"tại best epoch: val macro-F1 = {_fmt(s['val_macro_f1'])}, val acc = {_fmt(s['val_acc'])}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric, path: str, title: str = "", labels: list[str] | None = None) -> None:
    """Vẽ chồng một (hoặc nhiều) chỉ số của nhiều thí nghiệm, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    metric: tên khoá trong history ("val_loss", "val_macro_f1", "grad_norm", "epoch_time_s", ...),
            "gap" (= val_loss − train_loss), hoặc danh sách các tên đó (mỗi chỉ số một ô).
    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    labels = labels or [r["cfg"]["exp_id"] for r in results]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 4.2), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        values = []
        for r, lab in zip(results, labels):
            y = _series(r, m)
            values.append(y)
            ax.plot(r["history"]["epoch"], y, "o-", ms=2.5, label=lab + (" [DIVERGED]" if r["summary"].get("diverged") else ""))
        if m == "gap":
            ax.axhline(0, c="gray", lw=0.8)
        elif m not in ("val_acc", "val_macro_f1"):
            _auto_log(ax, values)
        ax.set(xlabel="epoch", ylabel=LABELS.get(m, m), title=LABELS.get(m, m))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_lr_sensitivity(results: list[dict], path: str, metric: str = "val_macro_f1", title: str = "") -> None:
    """Độ nhạy với lr: trục x = lr (log), trục y = chỉ số tại best epoch, mỗi bộ tối ưu một đường.
    Lần chạy bị phân kỳ được đánh dấu bằng dấu x ở đáy trục."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    groups = {}
    for r in results:
        groups.setdefault(r["cfg"]["optimizer"], []).append(r)
    for opt, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["cfg"]["lr"])
        lrs = [r["cfg"]["lr"] for r in rs]
        ys = [np.nan if r["summary"].get("diverged") or r["summary"][metric] is None else float(r["summary"][metric])
              for r in rs]
        line, = ax.plot(lrs, ys, "o-", label=OPT_NAMES.get(opt, opt))
        for lr, y in zip(lrs, ys):
            if np.isnan(y):
                ax.plot(lr, 0.0, "x", ms=10, c=line.get_color())
    ax.set_xscale("log")
    ax.set(xlabel="learning rate (log)", ylabel=LABELS.get(metric, metric) + " tại best epoch",
           title=title or "Độ nhạy với learning rate")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
