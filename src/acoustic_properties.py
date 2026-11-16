"""
acoustic_properties.py
======================
Thermophysical and acoustic properties of humid air, for the acoustic-streaming
PIV analysis (NACA 65(1)-412 / USound UT-P2020 campaign).

Main entry point
----------------
    acoustic_diffusivity(T_C, RH_pct, P_Pa, f_Hz=None, ...)

returns Lighthill's "diffusivity of sound"

    delta = nu * [ 4/3 + mu_B/mu + (gamma - 1)/Pr ]                    (1)
          = (1/rho0) * [ (4/3) mu + mu_B + kappa (gamma - 1)/c_p ]

which sets the bulk (free-space) attenuation of a plane wave:

    amplitude:  p ~ exp(-alpha x),   alpha = delta * omega^2 / (2 c0^3)
    power:      P ~ exp(-beta  x),   beta  = delta * omega^2 / c0^3   (= 2 alpha)

(beta is Lighthill's eq. 189 in Waves in Fluids, Sec. 4.7.)

Physics / what is and is not frequency independent
--------------------------------------------------
Eq. (1) has three pieces:

  (a) shear viscosity            4/3 * nu
  (b) heat conduction            nu * (gamma - 1) / Pr
  (c) bulk viscosity mu_B        nu * mu_B / mu

(a)+(b) is the classical Stokes-Kirchhoff result.  (c) represents the lag of
internal molecular degrees of freedom behind translation:

  * ROTATIONAL relaxation of N2/O2.  Relaxation time ~1e-9 s, so for any audio
    frequency it is a true, frequency-independent bulk viscosity,
    mu_B,rot ~ 0.60 mu  (Greenspan 1959).  Independent of humidity.

  * VIBRATIONAL relaxation of O2 and N2.  This is where humidity enters: water
    vapour catalyses the vibrational energy transfer, so the relaxation
    frequencies f_rO, f_rN depend strongly on the molar water concentration h.
    Each process behaves as a bulk viscosity ONLY for f << f_r.  In room air
    f_rN is a few hundred Hz, i.e. INSIDE our 30-500 Hz test range, so the
    "bulk viscosity" (and hence delta) is frequency dependent:

        delta_vib,i(f) = delta_vib,i(0) / (1 + (f / f_r,i)^2)

    Pass f_Hz to get the effective delta at the drive frequency; with
    f_Hz=None the low-frequency (equilibrium) limit is returned.

For room air the vibrational part is ~100x the classical part at a few hundred
Hz, so  delta_classical ~ 3.8e-5 m^2/s  is NOT a good estimate of the real
attenuation of audible sound in humid air.

Sources
-------
[L78]  Lighthill, M. J. (1978). Waves in Fluids. Cambridge Univ. Press.
       Sec. 1.13 (dissipation, definition of the diffusivity of sound delta),
       Sec. 4.7 eq. (189) (beta = delta omega^2 / c0^3).
       Form (1) of delta as quoted, attributing the name to Lighthill, in:
[J12]  Jordan, P. M., Norton, G. V., Chin-Bing, S. A., Warn-Varnas, A. (2012).
       "On the propagation of nonlinear acoustic waves in viscous and
       thermoviscous fluids." Eur. J. Mech. B/Fluids 34, 56-63, eq. (5).
       (Their primary reference: Lighthill, M. J. (1956), "Viscosity effects in
       sound waves of finite amplitude", in Surveys in Mechanics, CUP, 250-351.)
[P19]  Pierce, A. D. (2019). Acoustics: An Introduction to Its Physical
       Principles and Applications, 3rd ed., Springer. Ch. 10 (classical
       absorption, Sec. 10-2; relaxation and atmospheric absorption,
       Secs. 10-7, 10-8).  NOTE: Pierce's delta_cl is HALF of Lighthill's delta
       (Pierce writes alpha_cl = omega^2 delta_cl / c^3).
[K00]  Kinsler, Frey, Coppens, Sanders (2000). Fundamentals of Acoustics,
       4th ed., Wiley. Ch. 8 (classical absorption coefficient; molecular
       thermal relaxation).
[ISO]  ISO 9613-1:1993, Acoustics - Attenuation of sound during propagation
       outdoors - Part 1: Calculation of the absorption of sound by the
       atmosphere.  (Identical formulae: ANSI S1.26-1995.)  Source of the
       vibrational-relaxation formulae, f_rO, f_rN, h and p_sat used here.
[B95]  Bass, H. E., Sutherland, L. C., Zuckerwar, A. J., Blackstock, D. T.,
       Hester, D. M. (1995). "Atmospheric absorption of sound: Further
       developments." J. Acoust. Soc. Am. 97(1), 680-683 (+ erratum, JASA 99,
       1259 (1996)).  Physical basis of [ISO].
[G59]  Greenspan, M. (1959). "Rotational relaxation in nitrogen, oxygen, and
       air." J. Acoust. Soc. Am. 31(2), 155-160.  mu_B,rot ~ 0.6 mu for air.
       Cross-check: the [ISO] classical+rotational coefficient 1.84e-11 s^2/m
       implies mu_B/mu = 0.60 when combined with the transport properties
       below (see _self_check()).  Cramer (2012), Phys. Fluids 24, 066102,
       gives 0.77 for pure N2 at 300 K.
[W06]  White, F. M. (2006). Viscous Fluid Flow, 3rd ed., McGraw-Hill.
       Sutherland-law constants for mu(T) and kappa(T) of air.

VERIFICATION STATUS (2026-11): the formulae of [ISO] and eq. (5) of [J12] were
checked against web copies, and absorption_iso9613() reproduces the ISO
table values at 20 C (RH 10-50 %, 125 and 250 Hz) to within ~1 %
(src/tests/test_acoustic_properties.py).  [P19], [K00], [G59], [W06] are cited from memory
of the standard texts -- section/equation numbers should be confirmed against
a physical copy before they go in the paper.

Stated validity of [ISO]: accuracy about +/-10 % for 253-323 K, molar water
concentration 0.05-5 %, pressure < 200 kPa.

Simplifications
---------------
* mu and kappa are dry-air values (Sutherland).  At h ~ 1 % molar water the
  humid-air correction is < 1 % (Tsilingiris 2008, Energy Convers. Manage. 49,
  1098) -- well inside the +/-10 % of the relaxation model.
* Ideal-gas mixture for rho, c_p, gamma, c0.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "saturation_vapor_pressure",
    "molar_water_concentration",
    "relaxation_frequencies",
    "air_properties",
    "absorption_iso9613",
    "acoustic_diffusivity",
    "bulk_attenuation_coefficient",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
T_REF = 293.15          # K, reference temperature [ISO]
T_TRIPLE = 273.16       # K, triple-point isotherm temperature [ISO]
P_REF = 101325.0        # Pa, reference pressure [ISO]

R_UNIVERSAL = 8.314462618   # J/(mol K)
M_DRY_AIR = 28.9647e-3      # kg/mol
M_WATER = 18.01528e-3       # kg/mol
CP_DRY_AIR = 1005.0         # J/(kg K), 280-310 K
CP_WATER_VAPOR = 1865.0     # J/(kg K), ~300 K

# Sutherland constants for dry air [W06]
_MU_0, _T_MU_0, _S_MU = 1.716e-5, 273.15, 110.4       # Pa s, K, K
_K_0, _T_K_0, _S_K = 0.0241, 273.15, 194.0            # W/(m K), K, K

# Rotational-relaxation bulk viscosity of air, mu_B/mu [G59]; consistent with
# the 1.84e-11 coefficient of [ISO] (see _self_check).
MU_B_ROT_RATIO = 0.60

# Vibrational relaxation constants of [ISO] eq. (5): (coefficient, theta_vib [K])
_A_O2, _THETA_O2 = 0.01275, 2239.1
_A_N2, _THETA_N2 = 0.1068, 3352.0


# ---------------------------------------------------------------------------
# Humidity
# ---------------------------------------------------------------------------
def saturation_vapor_pressure(T_C):
    """Saturation vapour pressure of water [Pa] at temperature T_C [deg C].

    [ISO] Annex B:  p_sat/p_ref = 10**(-6.8346 (T01/T)**1.261 + 4.6151).
    """
    T = np.asarray(T_C, dtype=float) + 273.15
    return P_REF * 10.0 ** (-6.8346 * (T_TRIPLE / T) ** 1.261 + 4.6151)


def molar_water_concentration(T_C, RH_pct, P_Pa):
    """Molar concentration of water vapour h, in PERCENT ([ISO] Annex B).

    h = RH * p_sat / p_a   (RH in %, so h is in %).
    """
    return np.asarray(RH_pct, dtype=float) * saturation_vapor_pressure(T_C) / np.asarray(P_Pa, dtype=float)


def relaxation_frequencies(T_C, RH_pct, P_Pa):
    """Vibrational relaxation frequencies (f_rO, f_rN) of O2 and N2 in Hz.

    [ISO] eqs. (3), (4):
        f_rO = (pa/pr) {24 + 4.04e4 h (0.02 + h)/(0.391 + h)}
        f_rN = (pa/pr) (T/T0)^(-1/2) {9 + 280 h exp[-4.170 ((T/T0)^(-1/3) - 1)]}
    with h the molar water concentration in percent.
    """
    T = np.asarray(T_C, dtype=float) + 273.15
    pr = np.asarray(P_Pa, dtype=float) / P_REF
    h = molar_water_concentration(T_C, RH_pct, P_Pa)
    f_rO = pr * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    f_rN = pr * (T / T_REF) ** -0.5 * (
        9.0 + 280.0 * h * np.exp(-4.170 * ((T / T_REF) ** (-1.0 / 3.0) - 1.0))
    )
    return f_rO, f_rN


# ---------------------------------------------------------------------------
# Thermophysical properties
# ---------------------------------------------------------------------------
def air_properties(T_C=22.0, RH_pct=40.0, P_Pa=101325.0):
    """Thermophysical properties of humid air (ideal-gas mixture).

    Returns a dict with
        rho   density                      [kg/m^3]
        c0    sound speed (low-f, equil.)  [m/s]
        mu    shear viscosity              [Pa s]     (dry air, Sutherland [W06])
        nu    kinematic viscosity          [m^2/s]
        kappa thermal conductivity         [W/(m K)]  (dry air, Sutherland [W06])
        cp    specific heat, const. p      [J/(kg K)]
        gamma ratio of specific heats      [-]
        Pr    Prandtl number mu cp/kappa   [-]
        h     molar water concentration    [%]
        x_w   water mole fraction          [-]
    """
    T = np.asarray(T_C, dtype=float) + 273.15
    P = np.asarray(P_Pa, dtype=float)
    h = molar_water_concentration(T_C, RH_pct, P_Pa)
    x_w = h / 100.0

    M = (1.0 - x_w) * M_DRY_AIR + x_w * M_WATER           # kg/mol
    R_spec = R_UNIVERSAL / M                              # J/(kg K)
    w_w = x_w * M_WATER / M                               # water mass fraction
    cp = (1.0 - w_w) * CP_DRY_AIR + w_w * CP_WATER_VAPOR
    gamma = cp / (cp - R_spec)

    rho = P / (R_spec * T)
    c0 = np.sqrt(gamma * R_spec * T)

    mu = _MU_0 * (T / _T_MU_0) ** 1.5 * (_T_MU_0 + _S_MU) / (T + _S_MU)
    kappa = _K_0 * (T / _T_K_0) ** 1.5 * (_T_K_0 + _S_K) / (T + _S_K)

    return dict(rho=rho, c0=c0, mu=mu, nu=mu / rho, kappa=kappa, cp=cp,
                gamma=gamma, Pr=mu * cp / kappa, h=h, x_w=x_w)


# ---------------------------------------------------------------------------
# ISO 9613-1 absorption (reference implementation, used for cross-checks)
# ---------------------------------------------------------------------------
def absorption_iso9613(f_Hz, T_C=22.0, RH_pct=40.0, P_Pa=101325.0, components=False):
    """Pure-tone AMPLITUDE absorption coefficient alpha [Np/m], [ISO] eq. (5).

        alpha = f^2 { 1.84e-11 (pa/pr)^-1 (T/T0)^(1/2)
                      + (T/T0)^(-5/2) [ 0.01275 exp(-2239.1/T) / (f_rO + f^2/f_rO)
                                      + 0.1068  exp(-3352.0/T) / (f_rN + f^2/f_rN) ] }

    Multiply by 8.686 for dB/m.  With components=True returns the tuple
    (alpha_classical_plus_rotational, alpha_vib_O2, alpha_vib_N2).
    """
    f = np.asarray(f_Hz, dtype=float)
    T = np.asarray(T_C, dtype=float) + 273.15
    pr = np.asarray(P_Pa, dtype=float) / P_REF
    f_rO, f_rN = relaxation_frequencies(T_C, RH_pct, P_Pa)

    a_cl = f ** 2 * 1.84e-11 / pr * (T / T_REF) ** 0.5
    a_O = f ** 2 * (T / T_REF) ** -2.5 * _A_O2 * np.exp(-_THETA_O2 / T) / (f_rO + f ** 2 / f_rO)
    a_N = f ** 2 * (T / T_REF) ** -2.5 * _A_N2 * np.exp(-_THETA_N2 / T) / (f_rN + f ** 2 / f_rN)
    if components:
        return a_cl, a_O, a_N
    return a_cl + a_O + a_N


# ---------------------------------------------------------------------------
# Diffusivity of sound
# ---------------------------------------------------------------------------
def acoustic_diffusivity(T_C=22.0, RH_pct=40.0, P_Pa=101325.0, f_Hz=None, *,
                         include_vibrational=True,
                         mu_B_rot_ratio=MU_B_ROT_RATIO,
                         return_components=False):
    """Lighthill's diffusivity of sound, delta [m^2/s], for humid air.

        delta = nu [4/3 + mu_B/mu + (gamma - 1)/Pr]          ([L78] Sec. 1.13; [J12] eq. 5)
              = (1/rho0) [(4/3) mu + mu_B + kappa (gamma - 1)/c_p]

    so that a plane wave attenuates as  exp(-delta omega^2 x / (2 c0^3))  in
    amplitude, i.e.  beta = delta omega^2 / c0^3  in power ([L78] eq. 189).

    mu_B = mu_B,rot + mu_B,vib(f):
      * mu_B,rot = mu_B_rot_ratio * mu   (rotational relaxation [G59]; frequency
        and humidity independent at audio frequencies)
      * mu_B,vib(f) = effective bulk viscosity of the O2 and N2 vibrational
        relaxations, obtained by equating delta omega^2/(2 c0^3) to the
        relaxation absorption of [ISO] eq. (5):

            delta_vib,i(f) = (c0^3 / 2 pi^2) (T/T0)^(-5/2) A_i exp(-theta_i/T)
                             / ( f_r,i [1 + (f/f_r,i)^2] )

        This is where the RH (and most of the P) dependence comes from.

    Parameters
    ----------
    T_C : float or array      temperature [deg C]
    RH_pct : float or array   relative humidity [%]
    P_Pa : float or array     atmospheric pressure [Pa]
    f_Hz : float, array or None
        Acoustic frequency.  The vibrational contribution is frequency
        dependent; pass the drive frequency to get the effective delta there.
        None -> low-frequency (equilibrium) limit f << f_rN, f_rO, i.e. the
        only regime in which vibrational relaxation is a true bulk viscosity.
    include_vibrational : bool
        False -> classical thermoviscous + rotational only (the textbook
        "delta of air", ~3.8e-5 m^2/s; independent of f, ~independent of RH).
    mu_B_rot_ratio : float
        mu_B,rot / mu.  Default 0.60 [G59].
    return_components : bool
        True -> return a dict with the individual contributions.

    Returns
    -------
    delta : float or ndarray [m^2/s]   (or dict if return_components)

    Example
    -------
    >>> round(float(acoustic_diffusivity(22, 40, 101325, include_vibrational=False)) * 1e5, 2)
    3.82
    """
    pr = air_properties(T_C, RH_pct, P_Pa)
    nu, gamma, Pr, c0 = pr["nu"], pr["gamma"], pr["Pr"], pr["c0"]
    T = np.asarray(T_C, dtype=float) + 273.15

    d_shear = nu * 4.0 / 3.0
    d_thermal = nu * (gamma - 1.0) / Pr
    d_rot = nu * mu_B_rot_ratio

    f_rO, f_rN = relaxation_frequencies(T_C, RH_pct, P_Pa)
    f = 0.0 if f_Hz is None else np.asarray(f_Hz, dtype=float)
    pref = c0 ** 3 / (2.0 * np.pi ** 2) * (T / T_REF) ** -2.5
    d_vib_O = pref * _A_O2 * np.exp(-_THETA_O2 / T) / (f_rO * (1.0 + (f / f_rO) ** 2))
    d_vib_N = pref * _A_N2 * np.exp(-_THETA_N2 / T) / (f_rN * (1.0 + (f / f_rN) ** 2))
    if not include_vibrational:
        d_vib_O = d_vib_O * 0.0
        d_vib_N = d_vib_N * 0.0

    delta = d_shear + d_thermal + d_rot + d_vib_O + d_vib_N
    if return_components:
        return dict(delta=delta, shear=d_shear, thermal=d_thermal, rotational=d_rot,
                    vib_O2=d_vib_O, vib_N2=d_vib_N,
                    classical=d_shear + d_thermal,
                    classical_plus_rot=d_shear + d_thermal + d_rot,
                    mu_B_eff_over_mu=(d_rot + d_vib_O + d_vib_N) / nu,
                    f_rO=f_rO, f_rN=f_rN, **pr)
    return delta


def bulk_attenuation_coefficient(f_Hz, T_C=22.0, RH_pct=40.0, P_Pa=101325.0, **kwargs):
    """Bulk POWER attenuation coefficient beta = delta omega^2 / c0^3 [Np/m].

    [L78] eq. (189).  delta is evaluated at f_Hz (see acoustic_diffusivity);
    extra keyword arguments are passed through to acoustic_diffusivity, e.g.
    include_vibrational=False for the classical thermoviscous value.
    The amplitude coefficient is alpha = beta / 2.
    """
    f = np.asarray(f_Hz, dtype=float)
    c0 = air_properties(T_C, RH_pct, P_Pa)["c0"]
    delta = acoustic_diffusivity(T_C, RH_pct, P_Pa, f_Hz=f, **kwargs)
    return delta * (2.0 * np.pi * f) ** 2 / c0 ** 3


# ---------------------------------------------------------------------------
# Self-check / demo
# ---------------------------------------------------------------------------
def _self_check(verbose=True):
    """Consistency checks against [ISO]; raises AssertionError on failure."""
    # 1. delta (with vibrational terms) must reproduce the ISO absorption exactly
    #    apart from the classical term, which we build from first principles.
    f = np.array([30.0, 200.0, 500.0, 4000.0])
    for (T_C, RH) in [(20.0, 70.0), (22.0, 40.0), (10.0, 20.0)]:
        beta = bulk_attenuation_coefficient(f, T_C, RH, P_REF)
        iso = absorption_iso9613(f, T_C, RH, P_REF)
        assert np.allclose(beta / 2.0, iso, rtol=0.03), (T_C, RH, beta / 2 / iso)

    # 2. Classical + rotational part vs the ISO coefficient 1.84e-11 at 20 C, dry.
    p = air_properties(20.0, 0.0, P_REF)
    d_cl = acoustic_diffusivity(20.0, 0.0, P_REF, include_vibrational=False)
    coeff = 2.0 * np.pi ** 2 * d_cl / p["c0"] ** 3
    implied_ratio = (1.84e-11 * p["c0"] ** 3 / (2 * np.pi ** 2) / p["nu"]
                     - 4.0 / 3.0 - (p["gamma"] - 1.0) / p["Pr"])
    assert abs(coeff / 1.84e-11 - 1.0) < 0.02, coeff
    if verbose:
        print(f"[check] classical+rot coefficient: {coeff:.3e} s^2/m (ISO: 1.84e-11)")
        print(f"[check] mu_B,rot/mu implied by ISO coefficient: {implied_ratio:.3f}")
    return True


if __name__ == "__main__":
    _self_check()
    T_C, RH, P = 22.0, 40.0, 101325.0
    c = acoustic_diffusivity(T_C, RH, P, return_components=True)
    print(f"\nLab air: T = {T_C} C, RH = {RH} %, P = {P} Pa")
    for k in ("rho", "c0", "mu", "nu", "kappa", "cp", "gamma", "Pr", "h", "f_rO", "f_rN"):
        print(f"  {k:6s} = {float(c[k]):.5g}")
    print("\ndelta contributions, low-frequency limit [m^2/s]:")
    for k in ("shear", "thermal", "rotational", "vib_O2", "vib_N2", "delta"):
        print(f"  {k:11s} = {float(c[k]):.4e}")
    print("\n   f [Hz]   delta_eff [m^2/s]   beta [Np/m]   1/beta [km]")
    for f in (29.75, 57.76, 85.76, 113.76, 155.77, 197.77, 239.77, 505.8):
        d = float(acoustic_diffusivity(T_C, RH, P, f_Hz=f))
        b = float(bulk_attenuation_coefficient(f, T_C, RH, P))
        print(f"  {f:7.2f}   {d:.4e}          {b:.3e}     {1e-3 / b:.1f}")
