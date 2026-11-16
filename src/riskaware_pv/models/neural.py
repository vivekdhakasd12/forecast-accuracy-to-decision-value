"""Recurrent quantile network for day-ahead PV.

Design choices worth defending in the write-up:

1. **The sequence ends at the forecast origin.**  The encoder sees the 168
   hours up to and including gate closure and nothing after, so leakage is
   prevented by construction rather than by discipline.
2. **Monotone quantile head.**  Instead of predicting Q independent numbers and
   hoping they come out ordered, the head predicts a median and Q-1 *positive*
   increments (softplus), so the quantile function is non-decreasing by
   construction.  Crossing becomes impossible rather than merely rare.
3. **Pinball loss.**  Training and evaluation use the same proper scoring rule,
   so the reported CRPS is not measuring a train/test objective mismatch.
4. **Sequences are rebuilt on demand, never looked up by training origin.**  An
   earlier version keyed a sequence table by the origins seen during fitting;
   every test origin then missed the table and fell back to an all-zero week,
   so the encoder was fed nothing at inference while having trained on real
   history.  It still produced plausible numbers and simply scored worse than
   persistence, which is how it was caught.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # pragma: no cover - torch is optional for the rest of the study
    import os

    import torch
    import torch.nn as nn

    # Keep torch's pool aligned with the OpenMP cap set in riskaware_pv/__init__;
    # an unbounded pool alongside LightGBM's deadlocks the process on macOS.
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    nn = object  # type: ignore

from .base import Forecaster

SEQ_LEN = 168  # one week of hourly history ending at the origin


def _build_sequences(hist_y: pd.Series, hist_k: pd.Series,
                     origins: pd.DatetimeIndex, seq_len: int = SEQ_LEN
                     ) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """(n_unique_origins, seq_len, 2) history tensor ending at each origin.

    Every sampled timestamp lies in ``[origin - seq_len + 1, origin]``, so the
    encoder cannot see past gate closure regardless of what the source series
    contains -- the same guarantee the tabular features get from ``_gated_shift``.
    """
    uniq = pd.DatetimeIndex(sorted(set(origins)))
    offsets = pd.to_timedelta(np.arange(seq_len - 1, -1, -1), unit="h")
    stamps = uniq.values[:, None] - offsets.values[None, :]
    flat_index = pd.DatetimeIndex(stamps.ravel(), tz=uniq.tz)
    out = np.zeros((len(uniq), seq_len, 2), dtype=np.float32)
    for j, channel in enumerate((hist_y, hist_k)):
        vals = channel.reindex(flat_index).to_numpy(dtype=float)
        out[:, :, j] = np.nan_to_num(vals.reshape(len(uniq), seq_len), nan=0.0)
    return uniq, out


if _HAS_TORCH:

    class _QuantileLSTM(nn.Module):
        def __init__(self, n_static: int, n_quantiles: int, hidden: int = 64):
            super().__init__()
            self.encoder = nn.LSTM(2, hidden, num_layers=1, batch_first=True)
            self.static = nn.Sequential(
                nn.Linear(n_static, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, hidden), nn.ReLU(),
                nn.Linear(hidden, n_quantiles),
            )
            self.n_quantiles = n_quantiles
            self.median_idx = n_quantiles // 2

        def forward(self, seq, static):
            _, (h, _) = self.encoder(seq)
            z = torch.cat([h[-1], self.static(static)], dim=-1)
            raw = self.head(z)
            # Monotone parameterisation: a median plus cumulative positive steps.
            median = raw[:, self.median_idx: self.median_idx + 1]
            steps = torch.nn.functional.softplus(raw) + 1e-4
            below = -torch.flip(
                torch.cumsum(torch.flip(steps[:, : self.median_idx], [1]), 1), [1])
            above = torch.cumsum(steps[:, self.median_idx + 1:], 1)
            return torch.cat([median + below, median, median + above], dim=1)

    def pinball_loss(pred, target, quantiles):
        err = target.unsqueeze(1) - pred
        return torch.maximum(quantiles * err, (quantiles - 1.0) * err).mean()


class LSTMQuantile(Forecaster):
    """Sequence encoder + monotone quantile head."""

    name = "lstm_quantile"
    supports_native_quantiles = True

    def __init__(self, quantiles=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
                 hidden: int = 64, epochs: int = 25, batch_size: int = 256,
                 lr: float = 2e-3, seed: int = 0, device: str | None = None, **kw):
        super().__init__(**kw)
        if not _HAS_TORCH:
            raise ImportError("LSTMQuantile requires torch")
        qs = sorted(float(q) for q in quantiles)
        if len(qs) % 2 == 0:
            raise ValueError("supply an odd number of quantiles so the median is included")
        self.quantiles = tuple(qs)
        self.hidden, self.epochs, self.batch_size, self.lr = hidden, epochs, batch_size, lr
        self.seed = seed
        self.device = device or self._pick_device()

    @staticmethod
    def _pick_device() -> str:
        """CPU by default; an accelerator only when explicitly requested.

        ``mps.is_available()`` reports that the backend exists, not that memory
        is free.  On a unified-memory pool already under pressure, training
        passed the availability check and was then killed mid-run with no Python
        traceback.  This model is small and trains in a few seconds per epoch on
        CPU, so the accelerator is not worth that risk by default.  Set
        ``RISKAWARE_DEVICE=mps`` (or ``cuda``) to opt in.
        """
        requested = os.environ.get("RISKAWARE_DEVICE")
        if not requested:
            return "cpu"
        try:
            probe = torch.zeros(64, SEQ_LEN, 2, device=requested)
            del probe
            return requested
        except Exception:
            return "cpu"

    # ------------------------------------------------------------------ data
    Z_CLIP = 5.0  # standardised features are clipped to this many sigmas

    def _static_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Standardise, then clip.

        Gradient-boosted trees are invariant to feature scale and cope happily
        with a column that is entirely missing in training and present in test.
        A neural network does not.  The day-ahead price column illustrates the
        hazard: OPSD only reports the DE/LU series from late 2018, so in the
        first fold it is all-NaN during training -- giving a degenerate mean and
        a guarded unit variance -- and then arrives at test time as a z-score of
        128, which saturates the network and destroys the forecast.

        Columns that carry no usable training signal are dropped at fit time and
        whatever survives is clipped to +/- Z_CLIP, so an unseen shift in an
        input can distort a prediction but can no longer detonate it.
        """
        arr = X[self._cols].to_numpy(dtype=float)
        arr = np.where(np.isfinite(arr), arr, self._median)
        z = (arr - self._mu) / self._sigma
        return np.clip(z, -self.Z_CLIP, self.Z_CLIP).astype(np.float32)

    def _seq_for(self, X: pd.DataFrame) -> np.ndarray:
        """Encoder input for the origins in ``X``, built fresh from history."""
        origins = pd.DatetimeIndex(X["origin"])
        uniq, seq = _build_sequences(self._hist_y, self._hist_k, origins)
        pos = pd.Series(np.arange(len(uniq)), index=uniq).reindex(origins).to_numpy()
        out = seq[pos.astype(int)].copy()
        out[:, :, 0] /= max(self._seq_scale, 1e-6)
        return out

    def _store_history(self, hist: pd.DataFrame) -> None:
        y = pd.Series(hist["y"].to_numpy(dtype=float), index=hist.index).sort_index()
        k = pd.Series(hist["k"].to_numpy(dtype=float), index=hist.index).sort_index()
        self._hist_y = y[~y.index.duplicated()]
        self._hist_k = k[~k.index.duplicated()].fillna(0.0)

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y: np.ndarray,
            history: pd.DataFrame | None = None) -> "LSTMQuantile":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._cols = self._features(X)
        self._store_history(history if history is not None else X)

        self._seq_scale = 1.0
        _, probe = _build_sequences(self._hist_y, self._hist_k,
                                    pd.DatetimeIndex(X["origin"]))
        self._seq_scale = float(np.percentile(np.abs(probe[:, :, 0]), 99)) or 1.0

        # Keep only columns with enough finite training values to standardise.
        candidates = self._cols
        raw = X[candidates].to_numpy(dtype=float)
        finite_frac = np.isfinite(raw).mean(axis=0)
        with np.errstate(all="ignore"):
            spread = np.nanstd(np.where(np.isfinite(raw), raw, np.nan), axis=0)
        usable = (finite_frac > 0.5) & np.isfinite(spread) & (spread > 1e-8)
        self._cols = [c for c, keep in zip(candidates, usable) if keep]
        self._dropped_cols = [c for c, keep in zip(candidates, usable) if not keep]

        arr = X[self._cols].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            med = np.nanmedian(np.where(np.isfinite(arr), arr, np.nan), axis=0)
        self._median = np.nan_to_num(med, nan=0.0)
        arr = np.where(np.isfinite(arr), arr, self._median)
        self._mu = arr.mean(0)
        self._sigma = np.where(arr.std(0) < 1e-8, 1.0, arr.std(0))

        rows = self._train_rows(X)
        # Tensors stay on the CPU and move to the accelerator one minibatch at a
        # time; uploading a whole training set at once exhausted unified memory.
        static = torch.tensor(self._static_matrix(X)[rows])
        seq = torch.tensor(self._seq_for(X)[rows])
        z = torch.tensor(self._to_learning_target(X, y)[rows].astype(np.float32))
        qs = torch.tensor(self.quantiles, dtype=torch.float32, device=self.device)

        self._model = _QuantileLSTM(static.shape[1], len(self.quantiles),
                                    self.hidden).to(self.device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.epochs, 1))
        n = len(z)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            self._model.train()
            for i in range(0, n, self.batch_size):
                idx = perm[i: i + self.batch_size]
                opt.zero_grad()
                pred = self._model(seq[idx].to(self.device), static[idx].to(self.device))
                loss = pinball_loss(pred, z[idx].to(self.device), qs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                opt.step()
            sched.step()
        self._fitted = True
        return self

    # --------------------------------------------------------------- predict
    def _forward(self, X: pd.DataFrame) -> np.ndarray:
        self._model.eval()
        static = torch.tensor(self._static_matrix(X))
        seq = torch.tensor(self._seq_for(X))
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                chunk = self._model(seq[i: i + 2048].to(self.device),
                                    static[i: i + 2048].to(self.device))
                out.append(chunk.detach().cpu().numpy())
        return np.concatenate(out, axis=0)

    def predict_quantiles(self, X: pd.DataFrame, quantiles=None) -> np.ndarray:
        raw = self._forward(X)
        if quantiles is not None:
            want = np.asarray(quantiles, dtype=float)
            if len(want) != len(self.quantiles) or not np.allclose(want, self.quantiles):
                raw = np.vstack([np.interp(want, self.quantiles, row) for row in raw])
        return self._from_learning_target(X, raw)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._forward(X)[:, len(self.quantiles) // 2]
        return self._from_learning_target(X, raw)
