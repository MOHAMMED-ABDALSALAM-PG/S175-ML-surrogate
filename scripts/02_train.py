"""Train one component (regressor / clf1 / clf2) for one seed.

Run:
  python scripts/02_train.py --config configs/v3_masked_peroutput.yaml --seed 0
  python scripts/02_train.py --config ... --pilot          # timing only

`--pilot` measures optimiser throughput over a few hundred steps at several
batch sizes and exits without training. It exists to answer one question with
measurement instead of assumption: the original run took 103 hours for a single
seed, which makes a multi-seed study impossible unless that time comes down.
The suspected cause is in the original study's code -- BATCH_SIZE = 512 against 100.9M
rows is ~197,000 optimiser steps per epoch. The pilot measures what larger
batches and mixed precision actually buy, and the seed budget follows from the
measurement rather than from a guess.

Everything that affects a result is recorded next to the checkpoint: config,
resolved seed, git commit, environment, split checksums and dataset checksum.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from s175 import data as sdata
from s175.columns import (FEATURE_COLS, OUTPUT_COLS, FUEL_IDX, INFEASIBLE,
                          CLASS_VALID, CLASS_FUEL_ONLY, CLASS_ALL_NEG)
from s175.losses import MaskedAsymmetricMSE, build_alpha_vector, check_alpha_directions
from s175.metrics import binary_classifier
from s175.models import build_regressor, build_classifier
from s175.seeding import seed_everything
from s175.splits import load_split

RAW = Path("/home/macierz/mohabdal/S175_shaft_gen_off.txt")
CACHE_DIR = ROOT / "data" / "cache"
SPLIT_DIR = ROOT / "splits"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_commit() -> str:
    """The commit this run is attributable to, or "uncommitted" when there is none.

    `git rev-parse HEAD` in a repository with no commits exits non-zero and
    echoes the literal string "HEAD" on stdout, which the previous form
    recorded as the run's provenance. A run whose code is not committed must
    say so rather than name a hash-shaped non-hash.
    """
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True)
        head = p.stdout.strip()
        if p.returncode != 0 or len(head) != 40:
            return "uncommitted"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_commit(),
    }


class Standardiser:
    """Mean/std computed on the training split only.

    For the outputs the statistics are computed over contributing elements
    only: the -1 sentinel must never enter the mean or the variance, or the
    scaled target is corrupted for every row.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.where(std < 1e-8, 1.0, std).astype(np.float32)

    @classmethod
    def fit(cls, a: np.ndarray, mask: np.ndarray | None = None) -> "Standardiser":
        if mask is None:
            return cls(a.mean(axis=0), a.std(axis=0))
        m = mask.astype(np.float64)
        n = m.sum(axis=0)
        if (n == 0).any():
            raise ValueError("an output has no contributing rows for standardisation")
        mean = (a * m).sum(axis=0) / n
        var = (((a - mean) ** 2) * m).sum(axis=0) / n
        return cls(mean, np.sqrt(var))

    def transform(self, a: np.ndarray) -> np.ndarray:
        return ((a - self.mean) / self.std).astype(np.float32)

    def inverse(self, a: np.ndarray) -> np.ndarray:
        return (a * self.std + self.mean).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def load_arrays(split_name: str):
    arr, meta = sdata.load(RAW, cache_dir=CACHE_DIR)
    X, Y = sdata.split_xy(arr)
    cls = sdata.feasibility(Y, strict=False)
    sp = load_split(SPLIT_DIR, split_name)
    return X, Y, cls, sp, meta


def make_regression_tensors(X, Y, cls, idx, x_scaler, y_scaler):
    """Regression set: fully valid plus fuel-only rows, with the fuel mask."""
    keep = idx[(cls[idx] == CLASS_VALID) | (cls[idx] == CLASS_FUEL_ONLY)]
    xs = torch.from_numpy(x_scaler.transform(X[keep]))
    y_raw = Y[keep].copy()
    fuel_only = cls[keep] == CLASS_FUEL_ONLY
    mask = np.ones_like(y_raw, dtype=np.float32)
    mask[fuel_only, FUEL_IDX] = 0.0
    # Neutralise the sentinel before scaling so it cannot leak through the
    # scaled target; the mask makes its value irrelevant to the loss.
    y_raw[fuel_only, FUEL_IDX] = y_scaler.mean[FUEL_IDX]
    ys = torch.from_numpy(y_scaler.transform(y_raw))
    return xs, ys, torch.from_numpy(mask), len(keep)


def pilot(cfg, X, Y, cls, sp, device):
    """Measure optimiser throughput; do not train."""
    log("timing pilot -- measuring throughput, not training")
    x_scaler = Standardiser.fit(X[sp.train[: 5_000_000]])
    ymask = (cls[sp.train[: 5_000_000]] != CLASS_ALL_NEG)
    sub = sp.train[: 5_000_000][ymask]
    y_scaler = Standardiser.fit(Y[sub], mask=(Y[sub] != -1.0))

    xs, ys, mask, n = make_regression_tensors(X, Y, cls, sp.train[: 5_000_000],
                                              x_scaler, y_scaler)
    log(f"pilot pool: {n:,} rows")

    alpha = build_alpha_vector(cfg["alpha_speed"], cfg["alpha_others"])
    if not (cfg.get("original_loss", False) or cfg.get("ablation_loss", False)):
        check_alpha_directions(alpha)

    rows = []
    for bs in cfg.get("pilot_batch_sizes", [512, 4096, 16384, 65536]):
        for amp in ([False, True] if device.type == "cuda" else [False]):
            model = build_regressor(tuple(cfg["hidden"])).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
            loss_fn = MaskedAsymmetricMSE(alpha).to(device)
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            steps = cfg.get("pilot_steps", 200)

            xb_all = xs[: bs * 4].to(device)
            yb_all = ys[: bs * 4].to(device)
            mb_all = mask[: bs * 4].to(device)

            for w in range(10):  # warm-up
                s = (w % 4) * bs
                with torch.amp.autocast("cuda", enabled=amp):
                    l = loss_fn(model(xb_all[s:s+bs]), yb_all[s:s+bs], mb_all[s:s+bs])
                opt.zero_grad(set_to_none=True)
                scaler.scale(l).backward(); scaler.step(opt); scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for i in range(steps):
                s = (i % 4) * bs
                with torch.amp.autocast("cuda", enabled=amp):
                    l = loss_fn(model(xb_all[s:s+bs]), yb_all[s:s+bs], mb_all[s:s+bs])
                opt.zero_grad(set_to_none=True)
                scaler.scale(l).backward(); scaler.step(opt); scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            rps = steps * bs / dt
            n_train = int(len(sp.train))
            epoch_s = n_train / rps
            rows.append({"batch_size": bs, "amp": amp, "steps": steps,
                         "s_per_step": dt / steps, "rows_per_s": rps,
                         "projected_epoch_s": epoch_s,
                         "projected_epoch_h": epoch_s / 3600})
            log(f"  bs={bs:>6} amp={str(amp):<5} "
                f"{dt/steps*1000:7.2f} ms/step  {rps:12,.0f} rows/s  "
                f"projected epoch {epoch_s/3600:6.2f} h")
            del model, opt, xb_all, yb_all, mb_all
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out = ROOT / "results" / "timing_pilot.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"environment": environment(),
                               "n_train_rows": int(len(sp.train)),
                               "measurements": rows}, indent=2))
    log(f"wrote {out}")
    best = min(rows, key=lambda r: r["projected_epoch_s"])
    log(f"fastest: bs={best['batch_size']} amp={best['amp']} "
        f"-> {best['projected_epoch_h']:.2f} h/epoch")
    return 0


# ---------------------------------------------------------------------------
# full training
# ---------------------------------------------------------------------------
CHUNK = 4_000_000


def fit_standardiser(src: np.ndarray, idx: np.ndarray,
                     mask_sentinel: bool = False,
                     chunk: int = CHUNK) -> Standardiser:
    """Mean/std over `idx` rows of `src`, in two streaming passes.

    Streaming rather than `Standardiser.fit(src[idx])` because the training
    split is 100.9M rows: the array form widens both the data and the mask to
    float64 and multiplies them, peaking above 20 GB to produce statistics that
    fit in two vectors of ten numbers.

    mask_sentinel=True excludes the -1 sentinel elementwise, so an infeasible
    fuel entry never enters the mean or the variance -- the same guarantee
    `Standardiser.fit(..., mask=...)` gives, computed the same way.
    """
    n_cols = src.shape[1]
    total = np.zeros(n_cols, dtype=np.float64)
    count = np.zeros(n_cols, dtype=np.float64)
    for s in range(0, len(idx), chunk):
        a = src[idx[s:s + chunk]].astype(np.float64)
        if mask_sentinel:
            m = (a != INFEASIBLE).astype(np.float64)
            total += (a * m).sum(axis=0)
            count += m.sum(axis=0)
        else:
            total += a.sum(axis=0)
            count += len(a)
    if (count == 0).any():
        raise ValueError("a column has no contributing rows for standardisation")
    mean = total / count

    ss = np.zeros(n_cols, dtype=np.float64)
    for s in range(0, len(idx), chunk):
        a = src[idx[s:s + chunk]].astype(np.float64)
        d = (a - mean) ** 2
        if mask_sentinel:
            ss += (d * (a != INFEASIBLE)).sum(axis=0)
        else:
            ss += d.sum(axis=0)
    return Standardiser(mean, np.sqrt(ss / count))


def gather_scaled(src: np.ndarray, idx: np.ndarray,
                  scaler: Standardiser | None = None,
                  chunk: int = CHUNK) -> np.ndarray:
    """`scaler.transform(src[idx])` without holding the gather and its scaled
    copy at the same time."""
    out = np.empty((len(idx), src.shape[1]), dtype=np.float32)
    for s in range(0, len(idx), chunk):
        a = src[idx[s:s + chunk]]
        out[s:s + chunk] = scaler.transform(a) if scaler is not None else a
    return out


def build_regression_arrays(X, Y, cls, idx, x_scaler, y_scaler,
                            original_loss: bool = False):
    """Regression set: fully valid plus fuel-only rows, with the fuel mask.

    The array form of `make_regression_tensors`, gathered in chunks and sorted
    so the row gather runs forwards through memory rather than at random.

    original_loss=True reproduces the v1 baseline's data handling: fuel-only
    rows are dropped entirely (so no element is ever masked), exactly as
    step7_8_full.py did.
    """
    if original_loss:
        keep = idx[cls[idx] == CLASS_VALID]
    else:
        keep = idx[(cls[idx] == CLASS_VALID) | (cls[idx] == CLASS_FUEL_ONLY)]
    keep = np.sort(keep)
    xs = gather_scaled(X, keep, x_scaler)
    ys = gather_scaled(Y, keep, y_scaler)
    fuel_only = (cls[keep] == CLASS_FUEL_ONLY)
    # Neutralise the sentinel so it cannot leak through the scaled target. The
    # scaled value of the mean is exactly 0, which is what the unscaled form
    # (`y_raw[fuel_only, FUEL_IDX] = y_scaler.mean[FUEL_IDX]`) reduces to. The
    # mask makes the value irrelevant to the loss either way; this only keeps a
    # -1 out of the tensor.
    ys[fuel_only, FUEL_IDX] = 0.0
    return xs, ys, fuel_only


def build_classifier_arrays(X, cls, idx, x_scaler, positive, exclude=None):
    """Binary set for one classifier stage.

    `exclude` drops the rows the preceding stage has already caught, so C2 is
    trained on exactly the population it sees at inference: the rows C1 passed.
    """
    sel = idx if exclude is None else idx[cls[idx] != exclude]
    sel = np.sort(sel)
    xs = gather_scaled(X, sel, x_scaler)
    ys = (cls[sel] == positive).astype(np.float32)
    return xs, ys


def pick_batch_size(cfg: dict, args) -> tuple[int, bool]:
    """Batch size and AMP from the timing pilot, not from a guess.

    The config ships `batch_size: 512` only to record what the original run
    used. The pilot measured 512 at 51k rows/s and 65536 at 4.7M rows/s on this
    card, which is the difference between a 103-hour run and a one-hour one, so
    the measured optimum wins unless the caller overrides it explicitly.
    """
    if args.batch_size is not None:
        return args.batch_size, (cfg.get("amp", True) if args.amp is None else args.amp)
    p = ROOT / "results" / "timing_pilot.json"
    if p.exists():
        rows = json.loads(p.read_text()).get("measurements", [])
        if rows:
            best = min(rows, key=lambda r: r["projected_epoch_s"])
            amp = best["amp"] if args.amp is None else args.amp
            log(f"batch size {best['batch_size']} amp={amp} from the timing pilot "
                f"({best['rows_per_s']:,.0f} rows/s measured)")
            return int(best["batch_size"]), bool(amp)
    log(f"no timing pilot on disk; falling back to the config's batch_size "
        f"{cfg['batch_size']}")
    return int(cfg["batch_size"]), (cfg.get("amp", True) if args.amp is None else args.amp)


def store_device(nbytes: int, device: torch.device, headroom: float = 0.80):
    """Keep the training tensors on the GPU when they fit, else on the host.

    Resident data removes a host-to-device copy from every one of the ~1,300
    steps per epoch. The decision is made from the free memory the driver
    reports rather than assumed, and logged, so the same script degrades to
    streaming on a smaller card instead of dying with an OOM at epoch 1.
    """
    if device.type != "cuda":
        return torch.device("cpu"), False
    free, _total = torch.cuda.mem_get_info(device.index or 0)
    fits = nbytes < headroom * free
    log(f"  data {nbytes/2**30:.2f} GiB, {free/2**30:.2f} GiB free on the GPU "
        f"-> {'resident' if fits else 'streamed from host'}")
    return (device, True) if fits else (torch.device("cpu"), False)


def batch_order(n: int, bs: int, gen: torch.Generator, shuffle: bool,
                drop_last: bool, data_device: torch.device):
    """Batch index stream, permuted on the CPU and moved to the data.

    The permutation is always drawn from a CPU generator: torch.randperm gives
    a *different* sequence from a CPU and a CUDA generator under the same seed,
    and the tensors' home is decided at runtime from free VRAM -- seeding on
    the storage device would make the training trajectory depend on how busy
    the GPU happened to be.
    """
    order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    order = order.to(data_device)
    last = (n // bs) * bs if drop_last else n
    for s in range(0, last, bs):
        yield order[s:s + bs]


def run_split(model, loss_of, n: int, bs: int, gen, train: bool,
              opt=None, scaler=None, amp: bool = False,
              device: torch.device = None,
              data_device: torch.device = None) -> float:
    """One pass. Returns the loss summed over contributing elements / their count.

    Accumulating numerator and denominator separately, rather than averaging
    the per-batch means, keeps the reported figure exact when batches carry
    different numbers of contributing elements -- which they do, because the
    fuel mask removes a variable number per batch.
    """
    model.train(train)
    num = 0.0
    den = 0.0
    # drop_last only while training: a trailing batch of one row makes
    # BatchNorm's per-batch variance undefined. Re-shuffling each epoch means
    # no row is systematically excluded.
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for idx in batch_order(n, bs, gen, shuffle=train, drop_last=train,
                               data_device=data_device):
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                loss, weight = loss_of(idx)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            num += float(loss.detach()) * weight
            den += weight
    return num / den if den else float("nan")


def predict_probs(model, xs, n: int, bs: int, device, amp: bool) -> np.ndarray:
    """Sigmoid probabilities over a set, in evaluation order."""
    model.eval()
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for s in range(0, n, bs):
            xb = xs[s:s + bs].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logit = model(xb).squeeze(-1)
            out[s:s + bs] = torch.sigmoid(logit.float()).cpu().numpy()
    return out


def select_threshold(probs: np.ndarray, labels: np.ndarray,
                     target_recall: float) -> dict:
    """Choose the decision threshold on validation, never at an implicit 0.5.

    Positive class = infeasible, so a false negative is the dangerous error:
    the surrogate returns numbers where the simulator has none. Lowering the
    threshold raises recall, so the selection rule is the *highest* threshold
    that still meets the recall floor -- the most precise classifier that
    honours the safety constraint, rather than the most accurate one.

    Returns the chosen threshold, the rule that chose it, and the full sweep,
    so 04_evaluate can re-select under a different constraint without retraining.
    """
    if not labels.any():
        # Every recall in the sweep would be NaN and the selection would fall
        # through to an arbitrary grid point. A threshold chosen against zero
        # positive examples is meaningless; refuse rather than record one.
        raise ValueError(
            "validation split contains no positive rows -- cannot select a "
            "threshold. This happens on degenerate subsamples; the recorded "
            "splits all contain millions of positives."
        )
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    sweep = []
    for t in grid:
        # Strictly greater-than, matching Surrogate's `p > thr` at inference:
        # under `>=` a probability exactly at the threshold would count as
        # caught during selection but be missed at inference -- a discrepancy
        # in the unsafe direction.
        m = binary_classifier(labels, probs > t)
        sweep.append({"threshold": float(t), "recall": m["recall"],
                      "precision": m["precision"], "f1": m["f1"],
                      "false_negative_rate": m["false_negative_rate"]})
    ok = [s for s in sweep if s["recall"] >= target_recall]
    if ok:
        chosen = max(ok, key=lambda s: s["threshold"])
        rule = f"highest threshold with validation recall >= {target_recall}"
    else:
        best = max(sweep, key=lambda s: (s["f1"] if s["f1"] == s["f1"] else -1))
        chosen = best
        rule = (f"recall {target_recall} unreachable at any threshold "
                f"(best {max(s['recall'] for s in sweep):.4f}); fell back to max F1")
        log(f"  WARNING: {rule}")
    return {"threshold": chosen["threshold"], "rule": rule,
            "validation": chosen, "sweep": sweep}


def train_component(component: str, cfg: dict, args, X, Y, cls,
                    tr_idx: np.ndarray, va_idx: np.ndarray,
                    device: torch.device, bs: int, amp: bool,
                    x_scaler: Standardiser, y_scaler: Standardiser,
                    outdir: Path) -> dict:
    """Train one of the three networks and write its checkpoint and history."""
    log(f"--- {component} ---")
    n_out = len(OUTPUT_COLS)

    original_loss = bool(cfg.get("original_loss", False))
    t0 = time.perf_counter()
    if component == "regressor":
        xtr, ytr, ftr = build_regression_arrays(X, Y, cls, tr_idx, x_scaler, y_scaler,
                                                original_loss)
        xva, yva, fva = build_regression_arrays(X, Y, cls, va_idx, x_scaler, y_scaler,
                                                original_loss)
        nbytes = sum(a.nbytes for a in (xtr, ytr, xva, yva))
    else:
        positive = CLASS_ALL_NEG if component == "clf1" else CLASS_FUEL_ONLY
        exclude = None if component == "clf1" else CLASS_ALL_NEG
        xtr, ytr = build_classifier_arrays(X, cls, tr_idx, x_scaler, positive, exclude)
        xva, yva = build_classifier_arrays(X, cls, va_idx, x_scaler, positive, exclude)
        ftr = fva = None
        nbytes = sum(a.nbytes for a in (xtr, ytr, xva, yva))
    log(f"  built {len(xtr):,} train / {len(xva):,} val rows "
        f"in {time.perf_counter()-t0:.1f}s")

    # The previous component's tensors are unreachable by now, but the caching
    # allocator still holds their blocks and mem_get_info reports them as used,
    # so without this the second component would measure itself out of a card
    # that is in fact empty.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    sdev, resident = store_device(nbytes, device)
    to_store = lambda a: torch.from_numpy(a).to(sdev)
    xtr_t, ytr_t = to_store(xtr), to_store(ytr)
    xva_t, yva_t = to_store(xva), to_store(yva)
    ftr_t = to_store(ftr) if ftr is not None else None
    fva_t = to_store(fva) if fva is not None else None
    if resident:  # the host copies are dead once the GPU holds them
        del xtr, ytr, xva, yva, ftr, fva

    if component == "regressor":
        model = build_regressor(tuple(cfg["hidden"]), batchnorm=cfg["batchnorm"],
                                bn_after_act=cfg["bn_after_act"],
                                dropout=cfg["dropout"]).to(device)
        alpha = build_alpha_vector(cfg["alpha_speed"], cfg["alpha_others"])
        if original_loss:
            # v1 baseline: uniform alpha, deliberately violating SAFE_DIRECTION.
            # The guard exists to stop this happening by accident; the ablation
            # does it on purpose, so it must say so in the config.
            log("  original_loss: SAFE_DIRECTION check bypassed (v1 baseline)")
        elif cfg.get("ablation_loss", False):
            # loss-ablation arm: alphas deliberately violate SAFE_DIRECTION
            # (uniform or symmetric) while the fuel mask and standard data
            # handling are kept. Must be stated in the config, never implied.
            log("  ablation_loss: SAFE_DIRECTION check bypassed (loss ablation)")
        else:
            check_alpha_directions(alpha)   # the guard the original run lacked
        loss_fn = MaskedAsymmetricMSE(alpha).to(device)

        def make_loss(xs, ys, fo):
            def loss_of(idx):
                xb = xs[idx].to(device, non_blocking=True)
                yb = ys[idx].to(device, non_blocking=True)
                mask = torch.ones((len(idx), n_out), device=device)
                mask[fo[idx].to(device, non_blocking=True), FUEL_IDX] = 0.0
                return loss_fn(model(xb), yb, mask), float(mask.sum())
            return loss_of
        train_loss = make_loss(xtr_t, ytr_t, ftr_t)
        val_loss = make_loss(xva_t, yva_t, fva_t)
    else:
        hidden = tuple(cfg.get(f"hidden_{component}", cfg["hidden"]))
        model = build_classifier(hidden, batchnorm=cfg["batchnorm"],
                                 bn_after_act=cfg["bn_after_act"],
                                 dropout=cfg["dropout"]).to(device)
        bce = torch.nn.BCEWithLogitsLoss()

        def make_loss(xs, ys):
            def loss_of(idx):
                xb = xs[idx].to(device, non_blocking=True)
                yb = ys[idx].to(device, non_blocking=True)
                return bce(model(xb).squeeze(-1), yb), float(len(idx))
            return loss_of
        train_loss = make_loss(xtr_t, ytr_t)
        val_loss = make_loss(xva_t, yva_t)
    log(f"  {model.n_parameters():,} parameters")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    gscaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    max_epochs = cfg["max_epochs"] if args.max_epochs is None else args.max_epochs
    if max_epochs < 1:
        raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
    patience = cfg["patience"]

    n_tr, n_va = len(xtr_t), len(xva_t)
    # drop_last on the training pass means n_tr < bs yields *zero* batches: the
    # optimiser never steps, the loss is NaN, and a randomly-initialised
    # network would be checkpointed as a completed run. Refuse instead.
    if n_tr < bs:
        raise ValueError(
            f"{component}: {n_tr:,} training rows is fewer than the batch size "
            f"{bs:,}; no full batch can form. Pass a smaller --batch-size "
            f"(or a larger --subsample)."
        )
    best, best_epoch, best_state, waited = float("inf"), -1, None, 0
    history = []
    for epoch in range(max_epochs):
        # Seeded per epoch, so the shuffle is reproducible and independent of
        # the RNG stream the model's own randomness consumes. Always a CPU
        # generator -- see batch_order.
        gen = torch.Generator()
        gen.manual_seed(args.seed * 100_003 + epoch)
        e0 = time.perf_counter()
        tr = run_split(model, train_loss, n_tr, bs, gen, True, opt, gscaler,
                       amp, device, sdev)
        va = run_split(model, val_loss, n_va, bs, gen, False, amp=amp,
                       device=device, data_device=sdev)
        dt = time.perf_counter() - e0
        improved = va < best - 1e-12
        if improved:
            best, best_epoch, waited = va, epoch, 0
            best_state = {k: v.detach().to("cpu").clone()
                          for k, v in model.state_dict().items()}
        else:
            waited += 1
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va,
                        "seconds": dt, "improved": improved})
        log(f"  epoch {epoch:>3}  train {tr:.6f}  val {va:.6f}  "
            f"{dt:6.1f}s{'  *' if improved else f'  ({waited}/{patience})'}")
        if waited >= patience:
            log(f"  early stop: no improvement for {patience} epochs")
            break

    # History goes to disk before anything that can raise: on a diverged run
    # the epoch record is the evidence, and it must survive the failure.
    (outdir / f"{component}_history.json").write_text(json.dumps(history, indent=2))
    if best_state is None:
        raise RuntimeError(
            f"{component}: validation loss never improved over {len(history)} "
            f"epochs (NaN divergence?). No checkpoint written; history is in "
            f"{component}_history.json."
        )
    model.load_state_dict(best_state)
    result = {"component": component, "best_epoch": best_epoch,
              "best_val_loss": best, "epochs_run": len(history),
              "n_train_rows": int(n_tr), "n_val_rows": int(n_va),
              "n_parameters": int(model.n_parameters()),
              "batch_size": bs, "amp": bool(amp), "gpu_resident": bool(resident),
              "seconds": float(sum(h["seconds"] for h in history))}

    if component != "regressor":
        model.to(device)
        probs = predict_probs(model, xva_t, n_va, bs, device, amp)
        labels = (yva_t.to("cpu").numpy() > 0.5)
        result["threshold"] = select_threshold(
            probs, labels, cfg.get("clf_target_recall", 0.99))
        log(f"  threshold {result['threshold']['threshold']:.2f} "
            f"({result['threshold']['rule']}) -> "
            f"recall {result['threshold']['validation']['recall']:.4f} "
            f"precision {result['threshold']['validation']['precision']:.4f}")

    ckpt = {"component": component, "model": model.state_dict(),
            "model_config": model.config, "epoch": best_epoch,
            "x_scaler": x_scaler.state(),
            "y_scaler": y_scaler.state() if component == "regressor" else None,
            "threshold": result.get("threshold", {}).get("threshold")}
    torch.save(ckpt, outdir / f"{component}.pt")
    log(f"  best epoch {best_epoch} val {best:.6f}; wrote {component}.pt")
    return result


def train(cfg, args, X, Y, cls, sp, meta, split_name, device) -> int:
    bs, amp = pick_batch_size(cfg, args)
    # The split is part of the path: the run matrix trains the same config on
    # several holdout regimes, and without it seed_0 on S2 would silently
    # overwrite seed_0 on S1.
    outdir = ROOT / "runs" / cfg["name"] / split_name / f"seed_{args.seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    tr_idx, va_idx = sp.train, sp.val
    if args.subsample:
        # Applied here rather than inside the component, so the scalers are
        # fitted on the same rows the smoke run trains on. Strided rather than
        # sliced: the split indices are ascending, so a head slice would be one
        # contiguous corner of the lattice and could hold a single class.
        def thin(a, n):
            return a[:: max(1, len(a) // n)][:n]
        tr_idx = thin(tr_idx, args.subsample)
        va_idx = thin(va_idx, max(2, args.subsample // 8))
        log(f"SUBSAMPLED to {len(tr_idx):,} train / {len(va_idx):,} val rows "
            f"-- a smoke run, not a recorded one")

    # Scalers are fitted on the training split only, and the output statistics
    # exclude the completely-infeasible rows: they are -1 across the board and
    # would otherwise drag every output's mean toward the sentinel.
    t0 = time.perf_counter()
    x_scaler = fit_standardiser(X, np.sort(tr_idx))
    y_rows = np.sort(tr_idx[cls[tr_idx] != CLASS_ALL_NEG])
    y_scaler = fit_standardiser(Y, y_rows, mask_sentinel=True)
    log(f"scalers fitted on {len(tr_idx):,} train rows "
        f"({len(y_rows):,} contributing to the outputs) "
        f"in {time.perf_counter()-t0:.1f}s")

    wanted = (["regressor", "clf1", "clf2"] if args.component == "all"
              else [args.component])
    results = {}
    for c in wanted:
        results[c] = train_component(c, cfg, args, X, Y, cls, tr_idx, va_idx,
                                     device, bs, amp, x_scaler, y_scaler, outdir)

    split_meta = json.loads((SPLIT_DIR / split_name / "meta.json").read_text())
    run = {
        "config_path": str(args.config), "config": cfg, "seed": args.seed,
        "split": split_name, "split_sha256": split_meta["sha256"],
        "split_counts": split_meta["counts"],
        "dataset": {"source": meta.source, "sha256": meta.source_sha256,
                    "n_rows": meta.n_rows},
        "batch_size": bs, "amp": bool(amp), "device": str(device),
        "subsample": args.subsample, "environment": environment(),
        "components": results,
    }
    (outdir / "run.json").write_text(json.dumps(run, indent=2))
    log(f"wrote {outdir / 'run.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--component", default="all",
                    choices=["all", "regressor", "clf1", "clf2"])
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override the batch size the timing pilot selected")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override the config; for smoke runs")
    ap.add_argument("--subsample", type=int, default=None,
                    help="train on the first N rows of the split; smoke runs only")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    split_name = args.split or cfg.get("split", "S1_random")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The pilot measures throughput, so it runs without the deterministic
    # kernel constraint; recorded runs take the reproducibility over the speed.
    seed_everything(args.seed, deterministic=not args.pilot)
    log(f"config {args.config} | split {split_name} | seed {args.seed} | {device}")

    X, Y, cls, sp, meta = load_arrays(split_name)
    log(f"loaded {len(X):,} rows; split train {len(sp.train):,} "
        f"val {len(sp.val):,} test {len(sp.test):,}")

    if args.pilot:
        return pilot(cfg, X, Y, cls, sp, device)

    return train(cfg, args, X, Y, cls, sp, meta, split_name, device)


if __name__ == "__main__":
    raise SystemExit(main())
