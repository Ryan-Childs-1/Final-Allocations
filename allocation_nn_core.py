from __future__ import annotations
import json, math, os, time, zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

EPS = 1e-8

@dataclass
class MLPConfig:
    hidden: Tuple[int, ...]
    task: str = "regression"  # regression, binary, multi
    lr: float = 1e-3
    batch_size: int = 512
    epochs: int = 120
    dropout: float = 0.0
    weight_decay: float = 0.0
    patience: int = 20
    early_stopping: bool = True
    min_delta: float = 1e-5
    restore_best: bool = True
    seed: int = 42
    verbose_every: int = 5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def relu(x):
    return np.maximum(x, 0.0)


def d_relu(x):
    return (x > 0).astype(np.float32)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def safe_array(x, dtype=np.float32):
    arr = np.asarray(x, dtype=dtype)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def train_val_split(n: int, validation_frac: float = 0.16, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    val_n = max(1, int(n * validation_frac)) if n > 5 else max(0, n // 5)
    val = idx[:val_n]
    train = idx[val_n:]
    return train, val


class NumpyMLP:
    def __init__(self, input_dim: int, output_dim: int = 1, hidden: Tuple[int, ...] = (256,128), task: str = "regression", seed: int = 42):
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.task = task
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        dims = [self.input_dim] + list(self.hidden) + [self.output_dim]
        self.W, self.b = [], []
        for a, b in zip(dims[:-1], dims[1:]):
            scale = math.sqrt(2.0 / max(a, 1))
            self.W.append((self.rng.normal(0, scale, size=(a, b))).astype(np.float32))
            self.b.append(np.zeros((b,), dtype=np.float32))
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.step = 0
        self.history: List[Dict[str, Any]] = []
        self.early_stopping_summary: Dict[str, Any] = {}

    def forward(self, X, training=False, dropout=0.0):
        A = safe_array(X)
        pre, acts, masks = [], [A], []
        for i in range(len(self.W)-1):
            Z = A @ self.W[i] + self.b[i]
            A = relu(Z)
            if training and dropout > 0:
                mask = (self.rng.random(A.shape) >= dropout).astype(np.float32) / max(1.0 - dropout, EPS)
                A = A * mask
            else:
                mask = np.ones_like(A, dtype=np.float32)
            pre.append(Z); acts.append(A); masks.append(mask)
        Z = A @ self.W[-1] + self.b[-1]
        pre.append(Z)
        if self.task == "binary":
            out = sigmoid(Z)
        elif self.task == "positive_regression":
            out = softplus(Z)
        else:
            out = Z
        acts.append(out)
        return out, pre, acts, masks

    def predict(self, X):
        return self.forward(X, training=False)[0]

    def _loss_grad(self, pred, y):
        y = safe_array(y).reshape(pred.shape)
        if self.task == "binary":
            pred_clip = np.clip(pred, 1e-5, 1-1e-5)
            loss = -np.mean(y*np.log(pred_clip) + (1-y)*np.log(1-pred_clip))
            grad = (pred - y) / max(len(y), 1)
        else:
            diff = pred - y
            # Huber-ish for robust sizing/ranking
            delta = 1.0
            absd = np.abs(diff)
            loss = np.mean(np.where(absd <= delta, 0.5*diff*diff, delta*(absd - 0.5*delta)))
            grad = np.where(absd <= delta, diff, delta*np.sign(diff)) / max(len(y), 1)
            if self.task == "positive_regression":
                # derivative of softplus preactivation roughly sigmoid(output preactivation); use pred/(1+pred) approx
                grad = grad * np.clip(pred / (1.0 + pred), 0.01, 1.0)
        return float(loss), grad.astype(np.float32)

    def fit(self, X, y, X_val=None, y_val=None, config: Optional[MLPConfig] = None, name="model"):
        cfg = config or MLPConfig(hidden=self.hidden, task=self.task)
        X = safe_array(X); y = safe_array(y)
        if X_val is not None: X_val = safe_array(X_val)
        if y_val is not None: y_val = safe_array(y_val)
        n = len(X)
        best_val = float("inf"); best = None; wait = 0; best_epoch = 0
        has_val = X_val is not None and len(X_val) and y_val is not None and len(y_val)
        early_enabled = bool(cfg.early_stopping and has_val and cfg.patience and cfg.patience > 0)
        print(f"\n[{name}] train rows={n:,} val rows={0 if X_val is None else len(X_val):,} input_dim={self.input_dim} hidden={self.hidden} task={self.task}")
        print(f"[{name}] early_stopping={'ON' if early_enabled else 'OFF'} patience={cfg.patience} min_delta={cfg.min_delta:g} restore_best={cfg.restore_best}")
        for ep in range(1, cfg.epochs + 1):
            t0 = time.time()
            idx = np.arange(n)
            self.rng.shuffle(idx)
            train_losses = []
            for start in range(0, n, cfg.batch_size):
                bi = idx[start:start+cfg.batch_size]
                xb, yb = X[bi], y[bi]
                pred, pre, acts, masks = self.forward(xb, training=True, dropout=cfg.dropout)
                loss, grad = self._loss_grad(pred, yb)
                train_losses.append(loss)
                # backward
                dA = grad
                gW, gb = [None]*len(self.W), [None]*len(self.b)
                for layer in reversed(range(len(self.W))):
                    A_prev = acts[layer]
                    gW[layer] = A_prev.T @ dA + cfg.weight_decay * self.W[layer]
                    gb[layer] = dA.sum(axis=0)
                    if layer > 0:
                        dA = dA @ self.W[layer].T
                        dA = dA * d_relu(pre[layer-1]) * masks[layer-1]
                self.step += 1
                lr_t = cfg.lr * min(1.0, self.step / 100.0)
                beta1, beta2 = 0.9, 0.999
                for i in range(len(self.W)):
                    self.mW[i] = beta1*self.mW[i] + (1-beta1)*gW[i]
                    self.vW[i] = beta2*self.vW[i] + (1-beta2)*(gW[i]*gW[i])
                    self.mb[i] = beta1*self.mb[i] + (1-beta1)*gb[i]
                    self.vb[i] = beta2*self.vb[i] + (1-beta2)*(gb[i]*gb[i])
                    mw_hat = self.mW[i] / (1 - beta1**self.step)
                    vw_hat = self.vW[i] / (1 - beta2**self.step)
                    mb_hat = self.mb[i] / (1 - beta1**self.step)
                    vb_hat = self.vb[i] / (1 - beta2**self.step)
                    self.W[i] -= lr_t * mw_hat / (np.sqrt(vw_hat) + 1e-7)
                    self.b[i] -= lr_t * mb_hat / (np.sqrt(vb_hat) + 1e-7)
            train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
            val_loss = train_loss
            val_extra = {}
            if has_val:
                vp = self.predict(X_val)
                val_loss, _ = self._loss_grad(vp, y_val.reshape(vp.shape))
                if self.task == "binary":
                    yp = (vp.ravel() >= 0.5).astype(int); yt = y_val.ravel().astype(int)
                    tp = int(((yp==1)&(yt==1)).sum()); fp=int(((yp==1)&(yt==0)).sum()); fn=int(((yp==0)&(yt==1)).sum()); tn=int(((yp==0)&(yt==0)).sum())
                    val_extra = {"acc": float((tp+tn)/max(len(yt),1)), "precision": float(tp/max(tp+fp,1)), "recall": float(tp/max(tp+fn,1)), "f1": float(2*tp/max(2*tp+fp+fn,1))}
                else:
                    err = vp.ravel() - y_val.ravel()
                    val_extra = {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err*err))), "bias": float(err.sum())}
            row = {"epoch": ep, "train_loss": train_loss, "val_loss": float(val_loss), **val_extra, "seconds": round(time.time()-t0, 3)}
            self.history.append(row)
            if ep == 1 or ep % cfg.verbose_every == 0 or ep == cfg.epochs:
                extras = " ".join(f"{k}={v:.4f}" for k,v in val_extra.items() if isinstance(v, float))
                print(f"[{name}] epoch {ep:03d}/{cfg.epochs} train_loss={train_loss:.5f} val_loss={val_loss:.5f} {extras}")
            improved = float(val_loss) < (best_val - float(cfg.min_delta))
            row["early_stop_wait"] = int(wait)
            row["early_stop_best_val"] = float(best_val if np.isfinite(best_val) else val_loss)
            if improved or best is None:
                best_val = float(val_loss)
                best_epoch = int(ep)
                best = ([w.copy() for w in self.W], [b.copy() for b in self.b])
                wait = 0
                row["is_best"] = True
            else:
                wait += 1
                row["is_best"] = False
                if early_enabled and wait >= cfg.patience:
                    print(f"[{name}] early stopping at epoch {ep}; best_epoch={best_epoch}; best_val={best_val:.5f}; patience={cfg.patience}")
                    self.early_stopping_summary = {
                        "enabled": True,
                        "stopped_early": True,
                        "stopped_epoch": int(ep),
                        "best_epoch": int(best_epoch),
                        "best_val_loss": float(best_val),
                        "patience": int(cfg.patience),
                        "min_delta": float(cfg.min_delta),
                        "restore_best": bool(cfg.restore_best),
                    }
                    break
        else:
            self.early_stopping_summary = {
                "enabled": bool(early_enabled),
                "stopped_early": False,
                "stopped_epoch": int(cfg.epochs),
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val if np.isfinite(best_val) else np.nan),
                "patience": int(cfg.patience),
                "min_delta": float(cfg.min_delta),
                "restore_best": bool(cfg.restore_best),
            }
        if cfg.restore_best and best is not None:
            self.W, self.b = best
        if not self.early_stopping_summary:
            self.early_stopping_summary = {
                "enabled": bool(early_enabled),
                "stopped_early": False,
                "stopped_epoch": int(len(self.history)),
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val if np.isfinite(best_val) else np.nan),
                "patience": int(cfg.patience),
                "min_delta": float(cfg.min_delta),
                "restore_best": bool(cfg.restore_best),
            }
        return self

    def save(self, path: str, extra_meta: Optional[Dict[str, Any]] = None):
        meta = {"input_dim": self.input_dim, "output_dim": self.output_dim, "hidden": list(self.hidden), "task": self.task, "seed": self.seed, "history": self.history, "early_stopping": self.early_stopping_summary}
        if extra_meta: meta.update(extra_meta)
        arrays = {"meta": np.array(json.dumps(meta), dtype=object)}
        for i, w in enumerate(self.W): arrays[f"W{i}"] = w.astype(np.float32)
        for i, b in enumerate(self.b): arrays[f"b{i}"] = b.astype(np.float32)
        np.savez_compressed(path, **arrays)

    @staticmethod
    def load(path: str):
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"].item()))
        m = NumpyMLP(meta["input_dim"], meta["output_dim"], tuple(meta["hidden"]), meta["task"], meta.get("seed", 42))
        n = len(m.hidden) + 1
        m.W = [z[f"W{i}"].astype(np.float32) for i in range(n)]
        m.b = [z[f"b{i}"].astype(np.float32) for i in range(n)]
        m.history = meta.get("history", [])
        return m


def save_json(path: str, obj: Any):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else str(x))


def zip_files(zip_path: str, files: List[str], arc_prefix: str = ""):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            p = Path(f)
            if p.exists():
                z.write(p, arcname=str(Path(arc_prefix) / p.name))


def maybe_split_file(path: str, max_mb: float = 24.0):
    p = Path(path); size = p.stat().st_size / (1024*1024)
    if size <= max_mb:
        return {"file": p.name, "split": False, "size_mb": round(size, 3)}
    part_size = int(max_mb * 1024 * 1024)
    parts = []
    data = p.read_bytes()
    for i in range(0, len(data), part_size):
        part = p.with_name(f"{p.name}.part{i//part_size:03d}")
        part.write_bytes(data[i:i+part_size])
        parts.append(part.name)
    p.unlink()
    return {"file": p.name, "split": True, "parts": parts, "original_size_mb": round(size, 3), "max_part_mb": max_mb}
