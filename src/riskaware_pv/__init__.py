"""RiskAware-PV: probabilistic day-ahead PV forecasting and risk-aware dispatch.

Importing this package configures the OpenMP runtime, and that must happen
before NumPy, LightGBM or PyTorch are imported.

**Why this is here.**  On macOS, LightGBM links Homebrew's ``libomp`` while
PyTorch ships its own OpenMP runtime.  With two runtimes live in one process,
fitting a LightGBM model and then training a torch model *deadlocks*: the
process stops with no exception, no traceback and no output, which is close to
undiagnosable from a batch log.  The forecasting experiment fits both, so it
hung the moment the LSTM joined the model registry.

Measured on this stack: ``OMP_NUM_THREADS`` of 2 or 4 still deadlocks; only 1 is
safe.  The cost is small because LightGBM parallelises poorly on a design matrix
this shape -- five 500-tree quantile models take ~21 s single-threaded -- so the
serial setting buys reliability almost for free.

Override any of these from the environment if your platform does not need them.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

__version__ = "0.1.0"
