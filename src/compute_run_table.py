"""
compute_run_table.py
====================
Task 4: per-run scaling table for the interlaced speaker runs 2228-2239.

For each run, loads the 8 interlaced phases (PIVSpeakerInterlaced) and measures
  * U_rms        peak RMS of the coherent u_n oscillation near the orifice
                 (peak_rms(mode="u_n"): max of std-over-8-phases within
                 rho in [ETA_MIN, ETA_MAX], |X| <= 0.015, chord-normalised)
  * u_n_mean_pk  peak steady (streaming) u_n in the same window (for Task 5)
then computes, via streaming_scaling.run_table:
  f, U0_peak = sqrt(2) U_rms, P = 1/2 rho0 c0 U0^2 A_hole, delta(f), beta_bulk,
  F = P/c0, rho0 F/mu^2, Re, S, Re/S^2, forms_jet (Holman 2005, K = 0.16)
and writes data/run_table.csv.

Run in Colab (after the notebook's setup + import cells, Drive mounted):

    %run /content/NewCurve/src/compute_run_table.py
    # or with options:
    %run /content/NewCurve/src/compute_run_table.py --d-hole-mm 0.5 --T 22 --RH 40 --P 101325

Then download data/run_table.csv (from google.colab import files;
files.download("/content/NewCurve/data/run_table.csv")) and commit it.

ASSUMPTIONS TO CONFIRM (flagged in the CSV header comment as well)
  * d_hole = 0.5 mm  -- NOT confirmed (qualifying exam doesn't state it).
  * Lab air 22 C, 40 % RH, 101325 Pa -- placeholders, not logged values.
  * U0 is the near-wall PIV peak, not the exit velocity in the hole; it most
    likely underestimates U0 (see docs/task2_orifice_findings.md), so P, F,
    rho0 F/mu^2 and Re/S^2 are lower bounds.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import acoustic_properties as ap  # noqa: E402
import streaming_scaling as ss  # noqa: E402
from piv_speaker import PIVSpeakerInterlaced  # noqa: E402

RUNS = list(range(2228, 2240))
F_ARRAY = [505.8, 239.77, 197.77, 505.8, 239.77, 197.77,
           155.77, 113.76, 85.76, 57.76, 29.75, 239.77]   # Hz; repeats intentional
N_PHASES = 8
ETA_MIN, ETA_MAX = 0.0, 0.02        # chord-normalised search window (0-4 mm)
X_HALF = 0.015                      # |X| <= 0.015 c, as in peak_rms
CHORD = 0.2                         # m
DEFAULT_OUT = os.path.join(os.path.dirname(_SRC), "data", "run_table.csv")


def measure_run(run, rho0, eta_min=ETA_MIN, eta_max=ETA_MAX):
    """Load one interlaced run and return its near-orifice velocity measures."""
    I = PIVSpeakerInterlaced(run, N_PHASES, rho0=rho0)
    I.load_run()
    U_rms, rho_pk = I.peak_rms(etamin=eta_min, etamax=eta_max, mode="u_n", plot=False)

    window = np.all([eta_min <= I.rho, I.rho <= eta_max,
                     I.X <= X_HALF, I.X >= -X_HALF], axis=0)
    um = np.where(window, I.u_n_mean, np.nan)
    k = np.nanargmax(um)
    return dict(
        U_rms_m_s=float(U_rms),
        eta_U_rms_mm=float(rho_pk) * CHORD * 1e3,
        u_n_mean_peak_m_s=float(um.flat[k]),
        eta_u_n_mean_peak_mm=float(I.rho.flat[k]) * CHORD * 1e3,
    )


def build_table(runs, f, meas, d_hole, T_C, RH_pct, P_Pa):
    """Combine per-run measurements with the scaling quantities (pandas DataFrame)."""
    import pandas as pd

    ok = [i for i, m in enumerate(meas) if "error" not in m]
    rows = pd.DataFrame({"run": runs, "f_Hz": f})
    for key in ("U_rms_m_s", "eta_U_rms_mm", "u_n_mean_peak_m_s", "eta_u_n_mean_peak_mm"):
        rows[key] = [m.get(key, np.nan) for m in meas]
    rows["error"] = [m.get("error", "") for m in meas]

    if ok:
        sc = ss.run_table([runs[i] for i in ok], [f[i] for i in ok],
                          [meas[i]["U_rms_m_s"] for i in ok], d_hole,
                          T_C=T_C, RH_pct=RH_pct, P_Pa=P_Pa, U0_is_rms=True)
        sc = sc.rename(columns={"U0_m_s": "U0_peak_m_s"})
        sc["beta_bulk_classical_Np_m"] = ap.bulk_attenuation_coefficient(
            sc["f_Hz"].to_numpy(), T_C, RH_pct, P_Pa, include_vibrational=False)
        rows = rows.merge(sc.drop(columns="f_Hz"), on="run", how="left")

    cols = ["run", "f_Hz", "U_rms_m_s", "U0_peak_m_s", "eta_U_rms_mm",
            "u_n_mean_peak_m_s", "eta_u_n_mean_peak_mm",
            "P_W", "delta_m2_s", "beta_bulk_Np_m", "beta_bulk_classical_Np_m",
            "F_N", "rhoF_mu2", "Re", "S", "Re_over_S2", "forms_jet", "error"]
    for c in cols:
        if c not in rows:
            rows[c] = np.nan
    return rows[cols]


def write_csv(df, path, d_hole, T_C, RH_pct, P_Pa):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(f"# compute_run_table.py | d_hole = {d_hole*1e3:g} mm (UNCONFIRMED) | "
                 f"T = {T_C:g} C, RH = {RH_pct:g} %, P = {P_Pa:g} Pa (placeholders unless set) | "
                 f"U0_peak = sqrt(2) * peak u_n RMS within {ETA_MIN*CHORD*1e3:g}-"
                 f"{ETA_MAX*CHORD*1e3:g} mm of wall | Holman K = {ss.HOLMAN_K_AXISYMMETRIC}\n")
        df.to_csv(fh, index=False)


def print_table(df):
    show = df[["run", "f_Hz", "U0_peak_m_s", "u_n_mean_peak_m_s", "P_W",
               "beta_bulk_Np_m", "F_N", "rhoF_mu2", "Re", "S", "Re_over_S2",
               "forms_jet"]].copy()
    show.columns = ["run", "f [Hz]", "U0 [m/s]", "u_mean [m/s]", "P [W]",
                    "beta [Np/m]", "F [N]", "rhoF/mu2", "Re", "S", "Re/S2", "jet"]
    fmt = {c: "{:.3e}".format for c in ("U0 [m/s]", "u_mean [m/s]", "P [W]",
                                        "beta [Np/m]", "F [N]", "rhoF/mu2")}
    fmt.update({"f [Hz]": "{:.2f}".format, "Re": "{:.3g}".format,
                "S": "{:.3g}".format, "Re/S2": "{:.3g}".format})
    print(show.to_string(index=False, formatters=fmt, na_rep="--"))
    bad = df[df["error"] != ""]
    for _, r in bad.iterrows():
        print(f"  run {r['run']}: FAILED -- {r['error']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("--d-hole-mm", type=float, default=0.5, help="orifice diameter [mm] (unconfirmed)")
    p.add_argument("--T", type=float, default=22.0, help="lab temperature [C]")
    p.add_argument("--RH", type=float, default=40.0, help="relative humidity [%%]")
    p.add_argument("--P", type=float, default=101325.0, help="pressure [Pa]")
    p.add_argument("--out", default=DEFAULT_OUT, help="output CSV path")
    p.add_argument("--runs", type=int, nargs="*", help="subset of runs (default 2228-2239)")
    # Colab's %run passes the notebook's own argv; parse_known_args ignores it.
    args, _ = p.parse_known_args(argv)

    d_hole = args.d_hole_mm * 1e-3
    rho0 = float(ap.air_properties(args.T, args.RH, args.P)["rho"])
    runs, f = RUNS, F_ARRAY
    if args.runs:
        idx = [RUNS.index(r) for r in args.runs]
        runs, f = [RUNS[i] for i in idx], [F_ARRAY[i] for i in idx]

    meas = []
    for run, fr in zip(runs, f):
        print(f"run {run} ({fr} Hz) ...", flush=True)
        try:
            meas.append(measure_run(run, rho0))
        except Exception as exc:  # keep going; report at the end
            traceback.print_exc()
            meas.append({"error": f"{type(exc).__name__}: {exc}"})

    df = build_table(runs, f, meas, d_hole, args.T, args.RH, args.P)
    write_csv(df, args.out, d_hole, args.T, args.RH, args.P)
    print(f"\nd_hole = {args.d_hole_mm} mm (UNCONFIRMED), air: T = {args.T} C, "
          f"RH = {args.RH} %, P = {args.P} Pa, rho0 = {rho0:.4f} kg/m^3\n")
    print_table(df)
    print(f"\nwrote {os.path.abspath(args.out)}")
    return df


if __name__ == "__main__":
    main()
