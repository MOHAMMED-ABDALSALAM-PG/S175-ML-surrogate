"""Measure the S175 simulator's actual runtime, so the speed-up is measured
rather than estimated.

Run:  python scripts/06_timing_simulator.py [--repeats N] [--outdir DIR]

Background: the seminar slide quoted "~1-10 s per simulator call". That figure
was never measured -- it was an assumption -- and any speed-up ratio built on
it is unsupportable. This script measures the real cost on this machine.

Two costs are measured, because they answer different questions and differ by
orders of magnitude:

  single   One process invocation that evaluates one operating point
           (`-SafetyIndexes`, reading S175.wrin). Includes process start and
           the load of the hull/hydrostatic model data. This is what a caller
           pays to ask the simulator one question -- the relevant comparison for
           an optimiser that queries point by point.

  batch    `-CreateMetamodel` over a grid of K points in one invocation, so the
           fixed setup is paid once. Fitting total time against K separates the
           fixed setup from the marginal per-point cost. This is the fair
           comparison for bulk dataset generation, and it is the number that
           says what regenerating the 126M-row table would cost.

Everything runs in a private copy of the simulator directory, so the input
files in ~/S175_simulation/S175 are never modified.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

SRC = Path("/home/macierz/mohabdal/S175_simulation/S175")
BIN = "WeatherRouting.linux.bin"
PROJ = "S175_model_data.wrprj"

# A single operating point, well inside the feasible envelope.
WRIN = """Draft  = 8.7 m
Trim   = 0.0 m
GMt    = 0.55 m
Hs     = 3.72 m
Tp     = 10.73 s
Chim   = 2.793 rad
Vwind_rel  = 0.0 m/s
Thetaw_rel = 0.51 rad
EngRPM = 140.0 1/min
P/D = 0.8 no_unit
P_shaft_gen = 0.0 kW
"""


def build_metamod_params(n_chi: int, n_vwind: int = 1, n_theta: int = 1) -> tuple[str, int]:
    """Grid definition plus the exact number of points it generates.

    Only chi/Vwind/theta are scaled; the loading and propulsion parameters are
    held at one level each so the point count stays exactly predictable.
    """
    chi = [round(5.0 + i * 5.0, 3) for i in range(n_chi)]
    vwind = [round(i * 5.0, 3) for i in range(n_vwind)]
    theta = [round(i * 30.0, 3) for i in range(n_theta)]
    text = (
        "Draft = 8.7\n"
        "Trim  = 0.0\n"
        "EngRPM = 140.0\n"
        "P/D = 0.8\n"
        "P_shaft_gen = 0.0\n"
        "GMt = 0.55\n"
        "Hs_Tp = 3.72 10.73\n"
        f"Chi = {' '.join(str(v) for v in chi)}\n"
        f"Vwind = {' '.join(str(v) for v in vwind)}\n"
        f"Theta_wind = {' '.join(str(v) for v in theta)}\n"
    )
    return text, n_chi * n_vwind * n_theta


def run(cmd: list[str], cwd: Path) -> float:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} exited {p.returncode}:\n"
                           f"{p.stdout.decode(errors='replace')[-2000:]}")
    return dt


def count_data_rows(path: Path) -> int:
    """Data rows in a -FileOutput table, excluding the preamble."""
    n = 0
    with open(path, errors="replace") as f:
        for line in f:
            s = line.strip()
            if s and s[0].isdigit():
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10,
                    help="invocations per single-point measurement")
    ap.add_argument("--batch-repeats", type=int, default=3,
                    help="invocations per batch size; the median is used")
    ap.add_argument("--skip-single", action="store_true",
                    help="only run the batch ladder")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--work", default="/tmp/s175_timing",
                    help="where to place the private copy of the simulator")
    args = ap.parse_args()

    out = Path(args.outdir) if args.outdir else Path(__file__).resolve().parents[1] / "results"
    out.mkdir(parents=True, exist_ok=True)

    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    print(f"copying simulator to {work} (originals are never modified)", flush=True)
    shutil.copytree(SRC, work)
    (work / "S175.wrin").write_text(WRIN)
    binpath = work / BIN
    binpath.chmod(0o755)

    results: dict = {"host": subprocess.run(["hostname"], capture_output=True,
                                            text=True).stdout.strip()}

    # ---- single-point invocations ---------------------------------------
    if not args.skip_single:
        print(f"\n[1/2] single-point calls, {args.repeats} repeats", flush=True)
        cmd = [f"./{BIN}", PROJ, "S175.wrin", "-SafetyIndexes"]
        times = []
        for i in range(args.repeats):
            dt = run(cmd, work)
            times.append(dt)
            print(f"  call {i+1:>3}: {dt:8.3f} s", flush=True)
        results["single_point"] = {
            "n": len(times), "times_s": times,
            "mean_s": statistics.mean(times),
            "median_s": statistics.median(times),
            "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min_s": min(times), "max_s": max(times),
        }
        print(f"  mean {statistics.mean(times):.3f} s  "
              f"median {statistics.median(times):.3f} s", flush=True)
    else:
        prev = out / "simulator_timing.json"
        if prev.exists():
            results["single_point"] = json.loads(prev.read_text()).get("single_point")
            print("\n[1/2] single-point: carried over from the previous run",
                  flush=True)

    # ---- batch invocations ----------------------------------------------
    # A ladder of sizes with repeats, then a least-squares fit, because a
    # two-point fit is not defensible here: the per-point cost is not constant.
    # A feasible operating point is solved iteratively while an infeasible one
    # fails fast, so the mix of feasible and infeasible points in a grid
    # changes the average. The fit quality (R^2) is reported so the linearity
    # assumption is evidenced rather than asserted.
    print("\n[2/2] batch (-CreateMetamodel) at several sizes", flush=True)
    ladder = [(1, 1, 1), (12, 1, 1), (36, 1, 1), (36, 2, 1),
              (36, 3, 2), (36, 6, 2), (36, 6, 4)]
    batch = []
    for n_chi, n_v, n_t in ladder:
        text, expected = build_metamod_params(n_chi, n_v, n_t)
        (work / "S175_model_data" / "MetamodInputParams.dat").write_text(text)
        bcmd = [f"./{BIN}", PROJ, "S175.wrin", "-Concise", "-FileOutput",
                "-SafetyIndexes", "-CreateMetamodel"]
        reps = []
        for _ in range(args.batch_repeats):
            reps.append(run(bcmd, work))
        produced = count_data_rows(work / "output.txt")
        dt = statistics.median(reps)
        per = dt / produced if produced else float("nan")
        batch.append({"points_requested": expected, "points_produced": produced,
                      "times_s": reps, "median_total_s": dt, "per_point_s": per})
        print(f"  {produced:>6} points: {dt:8.3f} s median of {len(reps)}, "
              f"{per*1000:9.3f} ms/point", flush=True)

    results["batch"] = batch

    # Persisted before the fit. The ladder above costs several minutes of
    # simulator invocations; an exception in the analysis below must not throw
    # the measurements away.
    p = out / "simulator_timing.json"
    p.write_text(json.dumps(results, indent=2))

    xs = np.array([b["points_produced"] for b in batch], dtype=float)
    ys = np.array([b["median_total_s"] for b in batch], dtype=float)
    if len(xs) >= 3 and np.ptp(xs) > 0:
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = slope * xs + intercept
        ss_res = float(((ys - pred) ** 2).sum())
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        results["batch_fit"] = {
            "fixed_setup_s": float(intercept),
            "marginal_per_point_s": float(slope),
            "r2": r2,
            "note": ("least squares over the size ladder; per-point cost varies "
                     "with the feasible/infeasible mix of the grid, so this is "
                     "an average rather than a constant"),
        }
        print(f"\n  least-squares fit over {len(xs)} sizes:")
        print(f"    fixed setup    {intercept:8.3f} s")
        print(f"    marginal       {slope*1000:8.3f} ms/point")
        print(f"    fit R^2        {r2:8.5f}", flush=True)

        full = 126_153_720
        secs = intercept + slope * full
        print(f"\n  implied cost of regenerating all {full:,} rows: "
              f"{secs/3600:,.1f} h ({secs/86400:,.1f} days) single-threaded",
              flush=True)
        results["implied_full_dataset"] = {
            "n_rows": full, "seconds": secs, "hours": secs / 3600,
            "days": secs / 86400,
        }

    p.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
