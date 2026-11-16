"""
streaming_scaling.py
====================
Data-independent scaling quantities for the speaker / orifice streaming runs:
the Holman jet-formation criterion and the per-run table of Task 4.

Nothing here reads PIV data.  Feed it the measured velocity amplitude per run
(a_phase_array from PIVSpeakerInterlaced) and the drive frequencies.

References
----------
[H05]  Holman, R., Utturkar, Y., Mittal, R., Smith, B. L., Cattafesta, L.
       (2005). "Formation criterion for synthetic jets." AIAA J. 43(10),
       2110-2116.   Jet forms when  1/Sr = Re/S^2 > K,  K ~ 0.16 (axisymmetric
       orifice), K ~ 1 (2-D slot).
[L78]  Lighthill, M. J. (1978). Waves in Fluids, Sec. 4.7 (eqs. 189, 190-192,
       and the rho0 F / mu^2 regime parameter of fig. 84).

NOTE on velocity definitions (cited from memory -- check [H05] Sec. II before
this goes in the paper): Holman et al. build Re on U_bar, the exit velocity
averaged over the orifice area AND over the expulsion half-cycle.  For a
sinusoidal, spatially uniform exit velocity of amplitude U0,
U_bar = (2/pi) U0.  Pass velocity="amplitude" (default) if the U0 you supply
is the amplitude of the oscillation, and the conversion is applied; pass
velocity="holman" if it is already U_bar.
"""

from __future__ import annotations

import numpy as np

try:  # package or flat import (Colab: both files in the same folder)
    from . import acoustic_properties as ap
except ImportError:  # pragma: no cover
    import acoustic_properties as ap

HOLMAN_K_AXISYMMETRIC = 0.16   # [H05]
HOLMAN_K_2D = 1.0              # [H05]


def holman_criterion(f, U0, d_hole, nu, threshold=HOLMAN_K_AXISYMMETRIC,
                     velocity="amplitude"):
    """Synthetic-jet formation criterion of Holman et al. (2005).

        Re = U d / nu,   S = sqrt(omega d^2 / nu),   Re/S^2 = U / (omega d) = 1/Sr

    A jet (vortex rings escape the suction stroke) forms when Re/S^2 > threshold
    (0.16 for an axisymmetric orifice, ~1 for a 2-D slot).

    Parameters
    ----------
    f : array_like        drive frequency [Hz]
    U0 : array_like       exit velocity scale [m/s] (see `velocity`)
    d_hole : float        orifice diameter [m]
    nu : float            kinematic viscosity [m^2/s]
    threshold : float     formation constant K
    velocity : {"amplitude", "holman"}
        "amplitude": U0 is the amplitude of a sinusoidal exit velocity; it is
                     converted to Holman's U_bar = (2/pi) U0.
        "holman":    U0 is already U_bar.

    Returns
    -------
    dict of ndarrays: Re, S, Re_over_S2, forms_jet (bool), U_bar
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    U0 = np.atleast_1d(np.asarray(U0, dtype=float))
    if velocity == "amplitude":
        U_bar = 2.0 / np.pi * U0
    elif velocity == "holman":
        U_bar = U0
    else:
        raise ValueError("velocity must be 'amplitude' or 'holman'")
    omega = 2.0 * np.pi * f
    Re = U_bar * d_hole / nu
    S = np.sqrt(omega * d_hole ** 2 / nu)
    ratio = Re / S ** 2
    return dict(Re=Re, S=S, Re_over_S2=ratio, forms_jet=ratio > threshold, U_bar=U_bar)


def run_table(runs, f, U0, d_hole, T_C=22.0, RH_pct=40.0, P_Pa=101325.0,
              U0_is_rms=False, threshold=HOLMAN_K_AXISYMMETRIC):
    """Per-run scaling table (Task 4).

    Parameters
    ----------
    runs : sequence       run identifiers
    f : array_like        drive frequency per run [Hz]
    U0 : array_like       measured coherent velocity per run [m/s]
    d_hole : float        orifice diameter [m]
    T_C, RH_pct, P_Pa     lab conditions (set rho0, c0, mu, nu, delta)
    U0_is_rms : bool      True if U0 is the RMS of the oscillation (e.g. the
                          std over the 8 phases); it is then multiplied by
                          sqrt(2) to get the amplitude used below.

    Columns
    -------
    f_Hz, U0_m_s (amplitude),
    P_W          = 1/2 rho0 c0 U0^2 A_hole           (plane-wave acoustic power)
    delta_m2_s   = effective diffusivity of sound at f (acoustic_properties)
    beta_bulk_Np_m = delta omega^2 / c0^3             ([L78] eq. 189)
    F_N          = P / c0 = 1/2 rho0 U0^2 A_hole
    rhoF_mu2     = rho0 F / mu^2                      ([L78] fig. 84 parameter)
    Re, S, Re_over_S2, forms_jet                      (holman_criterion)

    Returns a pandas.DataFrame.
    """
    import pandas as pd

    f = np.atleast_1d(np.asarray(f, dtype=float))
    U0 = np.atleast_1d(np.asarray(U0, dtype=float))
    if not (len(runs) == f.size == U0.size):
        raise ValueError("runs, f and U0 must have the same length")
    if U0_is_rms:
        U0 = np.sqrt(2.0) * U0

    air = ap.air_properties(T_C, RH_pct, P_Pa)
    rho, c0, mu, nu = (float(air[k]) for k in ("rho", "c0", "mu", "nu"))
    A = np.pi * d_hole ** 2 / 4.0

    P = 0.5 * rho * c0 * U0 ** 2 * A
    F = P / c0
    delta = ap.acoustic_diffusivity(T_C, RH_pct, P_Pa, f_Hz=f)
    beta = ap.bulk_attenuation_coefficient(f, T_C, RH_pct, P_Pa)
    h = holman_criterion(f, U0, d_hole, nu, threshold=threshold, velocity="amplitude")

    return pd.DataFrame({
        "run": list(runs),
        "f_Hz": f,
        "U0_m_s": U0,
        "P_W": P,
        "delta_m2_s": delta,
        "beta_bulk_Np_m": beta,
        "F_N": F,
        "rhoF_mu2": rho * F / mu ** 2,
        "Re": h["Re"],
        "S": h["S"],
        "Re_over_S2": h["Re_over_S2"],
        "forms_jet": h["forms_jet"],
    })
