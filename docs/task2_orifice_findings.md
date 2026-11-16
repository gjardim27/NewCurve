# Task 2: orifice diameter: what the qualifying exam says

Source: `Schenkman_Qualifying_Examination.pdf` (74 pp.). Page numbers below are
the printed page numbers.

## Hole diameter: not stated

- §2.1.2 (p. 23) and §2.1 (p. 21): the speakers sit "slightly beneath the
  surface, with a small, corresponding hole in the suction surface above".
  The PLA suction surfaces were 3D-printed (Raise3D N2 Plus) with "precisely
  positioned holes". **No diameter is given.**
- Fig. 2.2 (p. 24): the holes are drawn "enlarged for visibility", so the
  figure can't be used to measure them.
- Fig. 5.1 (p. 49) and the Ch. 5 intro (p. 48): the SPL measurements were weak
  "due to the small size of the actuator holes compared with the microphone
  diameter". This only says the hole is smaller than the microphone, and the
  microphone model isn't named. SPL drops below measurable levels around
  100 Hz.
- §2.1.6 (p. 28): USound UT-P2020 MEMS speaker, 28 Vpp drive (amplifier
  saturation limit), USound UA-R3010 amplifier with a ~30 Hz high-pass
  response, Agilent 33210A function generator. The speaker datasheet is ref.
  [75] (USound GmbH, Achelous UT-P2020 datasheet, 2021). It would give the
  speaker's own sound-port size, but not the wing-hole diameter.

**Ask Ari:** the hole diameter in the suction-surface CAD (and whether there is
a cavity or neck between the speaker port and the surface). The UT-P2020
datasheet would also help; add it to the repo if you have it.

## Other things found that affect the brief

1. **Measured velocity scale vs. U0 ~ 1 m/s.** Fig. 5.4 (p. 52, 240 Hz): the
   colour scale for both the mean and the RMS of u_n is ±8 mm/s, and the
   near-orifice values are at the top of that scale. Text p. 48 estimates
   u'_p = O(1e-3) U_inf. So the coherent velocity *at the first PIV vectors*
   is O(1 cm/s), about 100x below the brief's U0_peak ~ 1 m/s. Whether the
   exit velocity *in* the hole is much larger can't be read from the PDF.
   Config. B resolution is 0.1 mm (Table 2.2), so the hole itself is barely
   resolved.
   Consequences (d = 0.5 mm, f = 240 Hz, rho0 = 1.19):

   | U0_peak | F = ½ rho0 U0² A | rho0 F / mu² | Re/S² (Holman, U_bar = 2U0/pi) |
   |---|---|---|---|
   | 1 m/s (brief) | 1.2e-7 N | ~420 | 0.84, so a jet forms |
   | 1 cm/s (Fig. 5.4 scale) | 1.2e-11 N | ~0.04 | 0.008, so no jet is predicted |

   At 240 Hz with d = 0.5 mm, Holman's criterion needs U0_peak > ~0.19 m/s.
2. **Field of view.** Table 2.2: Config. B FOV is 2 cm (0.1c), cropped further.
   Figs. 5.3 to 5.5 span ±4 mm in X and 0 to 8 mm in Y. The brief's "~4 cm
   FOV" is too large. A smaller FOV only strengthens the conclusion that bulk
   attenuation is negligible.
3. **RMS convention confirmed.** p. 54: "the STD of 8 samples spaced by π/4 of
   a sinusoid is identically equal to its RMS amplitude", and eq. (5.5) uses
   c(f_e) = √2 ũ'/A_e. This matches the brief's U0_peak = √2 × RMS.
4. **Frequency dependence.** p. 50: the streaming jet is "not present for lower
   frequencies, and is significantly stronger at f_e = 500 Hz". Ari cites
   Lighthill pp. 339–344 (force ∝ P f_e²). Relevant to Task 5.
5. **Section map for the methods chunks** (the brief has §2.1.6 as Config B,
   but it isn't):
   - §2.1.4 *PIV Configurations*: Config. A and **Config. B** (pp. 26–28, Table 2.2)
   - §2.1.6 *Embedded Actuators*: UT-P2020, amplifier, gain look-up table (pp. 28–29)
   - §2.2.2 *Particle Image Velocimetry* post-processing (p. 32)
   - §5.1 *PIV Measurements*: the interlaced phase technique, eq. (5.1) (pp. 48–50)
6. **The 0.1 % PTU frequency adjustment** isn't mentioned anywhere in the PDF.
   It only says that a "LaVision precision timing unit" syncs the camera and laser
   (p. 27). That paragraph will need Ari's input.
