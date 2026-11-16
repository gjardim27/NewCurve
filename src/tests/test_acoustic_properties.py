"""
Quick checks and printout for src/acoustic_properties.py.

Run either way (from the repo root):
    python src/tests/test_acoustic_properties.py     # prints a table
    python -m pytest src/tests                       # runs the asserts
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import acoustic_properties as ap  # noqa: E402

LAB = (22.0, 40.0, 101325.0)            # T [C], RH [%], P [Pa]
BRIEF_DELTA = 4.5e-5                    # m^2/s, approximate value in the project brief
RUN_FREQS = (29.75, 57.76, 85.76, 113.76, 155.77, 197.77, 239.77, 505.8)

# Independent reference: ISO 9613-1 attenuation at 20 C, 1 atm, in dB/km, as
# tabulated on Wikibooks "Engineering Acoustics/Outdoor Sound Propagation"
# (retrieved 2026-11-15; secondary source, check against ISO before the paper).
ISO_TABLE_20C = {  # RH: (125 Hz, 250 Hz)
    10: (0.775, 1.58),
    20: (0.712, 1.39),
    30: (0.615, 1.42),
    50: (0.445, 1.32),
}


def test_self_check():
    assert ap._self_check(verbose=False)


def test_iso_table():
    for rh, (a125, a250) in ISO_TABLE_20C.items():
        ours = ap.absorption_iso9613(np.array([125.0, 250.0]), 20.0, rh, 101325.0) * 8.686e3
        assert np.allclose(ours, [a125, a250], rtol=0.03), (rh, ours)


def test_classical_delta_near_textbook():
    # shear + heat conduction + rotational bulk viscosity only
    d = float(ap.acoustic_diffusivity(*LAB, include_vibrational=False))
    assert 3.5e-5 < d < 4.2e-5, d


def test_vibrational_relaxation_dominates_at_audio_f():
    c = ap.acoustic_diffusivity(*LAB, f_Hz=200.0, return_components=True)
    assert float(c["delta"]) > 50 * float(c["classical_plus_rot"])


def _print_table():
    print("delta [m^2/s] for a few conditions")
    print(f"{'T [C]':>6} {'RH [%]':>7} {'P [kPa]':>8} {'f [Hz]':>8} "
          f"{'classical+rot':>14} {'full delta':>11} {'full/brief':>11}")
    for T, RH, P in [LAB, (20.0, 50.0, 101325.0), (22.0, 20.0, 101325.0),
                     (22.0, 70.0, 101325.0), (30.0, 40.0, 101325.0),
                     (22.0, 40.0, 98000.0)]:
        for f in (None, 200.0, 505.8):
            dc = float(ap.acoustic_diffusivity(T, RH, P, include_vibrational=False))
            d = float(ap.acoustic_diffusivity(T, RH, P, f_Hz=f))
            fs = "f->0" if f is None else f"{f:.1f}"
            print(f"{T:6.1f} {RH:7.1f} {P/1e3:8.2f} {fs:>8} {dc:14.3e} {d:11.3e} {d/BRIEF_DELTA:11.1f}")

    print("\nLab air (22 C, 40 %, 101325 Pa), bulk POWER attenuation at the run frequencies")
    print(f"{'f [Hz]':>8} {'beta full [Np/m]':>17} {'beta classical [Np/m]':>22}")
    for f in RUN_FREQS:
        b = float(ap.bulk_attenuation_coefficient(f, *LAB))
        bc = float(ap.bulk_attenuation_coefficient(f, *LAB, include_vibrational=False))
        print(f"{f:8.2f} {b:17.3e} {bc:22.3e}")


if __name__ == "__main__":
    test_self_check()
    test_iso_table()
    test_classical_delta_near_textbook()
    test_vibrational_relaxation_dominates_at_audio_f()
    print("all checks passed\n")
    _print_table()
