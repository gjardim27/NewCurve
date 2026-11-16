# Acoustic Streaming PIV Paper — Project Recap

*Prepared for meeting with advisor, September 10, 2026*

---

## 1. Project Overview

You're co-authoring a paper on **PIV measurements of a small speaker flush-mounted in a wall**, specifically in the surface of a NACA 651412 airfoil model in the wind tunnel. The paper has two intertwined threads: an **experimental/methodological** contribution (largely your advisor's) and a **theory + analysis** contribution (largely yours).

### The experimental technique (advisor's contribution)

Standard phase-locked PIV samples the same phase of the actuation cycle every N cycles. The problem: unsteady background flow in the wind tunnel drifts between phases, so you can never cleanly separate speaker-driven flow from background flow. Your advisor modified this: instead of hitting the same phase every time, **add a 45° phase lag to each successive capture**. After eight captures you're back to the starting phase. This gives you a stroboscopic, quasi-time-resolved view of the periodic flow field across a full cycle.

The two things this technique cleanly separates:
- The **mean of the 8 phase averages = the steady acoustic streaming field** (this is the physically interesting quantity).
- **Deviations from that mean at each phase = the oscillating speaker component.**

That decomposition is what makes the whole analysis possible. Without it, streaming and oscillation are conflated.

### Your role

Because the technique itself is your advisor's, your writing contribution focuses on:
1. The **acoustic streaming theory and analysis section** (this is the intellectual core of your contribution).
2. **Drafting materials/methods in LaTeX** once your advisor creates the document.
3. Longer-term (fall): **designing and running wind-on phase-locked PIV** using weak forcing to lock in shedding frequency — since your advisor won't be active in the lab next year, this is your project to lead.

---

## 2. The Physics — Deep Dive

### Why acoustic streaming happens (Lighthill's framework)

Acoustic streaming is a **steady mean flow driven by an oscillating acoustic field**. The mechanism, in Lighthill's clean formulation (*Waves in Fluids* §4.7), is this:

An acoustic wave carries momentum. The time-averaged momentum flux appears mathematically as a **Reynolds stress**, $\overline{\rho_0 u_i u_j}$. The effective mean force per unit volume on the fluid is the **divergence of that stress**:

$$F_j = -\frac{\partial}{\partial x_i}\left(\overline{\rho_0 u_i u_j}\right) \quad \text{(Lighthill eq. 182)}$$

**Key subtlety — why does attenuation matter?** If the wave is unattenuated, the Reynolds stress is uniform along the beam. Uniform stress has no divergence in the streamwise direction, and any transverse gradient is exactly balanced by a mean pressure gradient. Net result: **no streaming**. You need the wave to *lose amplitude* somewhere for the stress to have a nonzero net gradient. That's the whole story: **streaming = attenuation deposits momentum unevenly in space.**

### The two attenuation mechanisms

There are two ways an acoustic wave loses amplitude in your setup:

**Bulk attenuation** (Lighthill eq. 189) from viscous and thermal dissipation in the free stream:
$$\beta_{\text{bulk}} = \frac{\delta \omega^2}{c_0^3}$$
where $\delta \approx 4.5 \times 10^{-5}$ m²/s is the diffusivity of sound. At 200 Hz this gives $\beta \sim 10^{-7}$ Np/m — meaning the wave decays over ~10⁷ meters. **Utterly negligible** across your ~4 cm field of view.

**Boundary-layer attenuation** in the orifice (Lighthill eq. 201, though derived for a tube):
$$\beta_{\text{BL}} = \frac{\sqrt{2\nu\omega}}{c_0 a}$$
where $a \approx 0.25$ mm is the orifice radius. At 200 Hz this gives $\beta \sim 2.4$ Np/m — **about $10^7$ times larger** than bulk attenuation.

**This is your key insight, and the one you presented at the May 20 meeting.** All the meaningful attenuation happens inside the orifice. Once the fluid emerges into the free stream, further attenuation is negligible. The streaming force is therefore *not distributed along the beam*, as it would be for the free-space case Lighthill mostly treats — **it's concentrated at the orifice.**

### From attenuation to a concentrated force

Because attenuation is localized, the whole speaker acts like a **single concentrated force F applied at the orifice**, ejecting fluid into the tunnel. This is the framing your advisor pushed you toward: skip the beam-integrated formulas, treat the source as a point force, and use Lighthill's stokeslet solutions (eqs. 195–197) to describe the resulting jet.

**Estimating F.** The acoustic power radiated by the speaker is:
$$P = \tfrac{1}{2} \rho_0 c_0 U_0^2 A_{\text{hole}}$$
where $U_0$ is the peak oscillatory velocity in the orifice (from your `a_phase_array`). The corresponding radiation force is $F = P/c_0$:
$$\boxed{F = \tfrac{1}{2} \rho_0 U_0^2 A_{\text{hole}}}$$

For representative numbers ($U_0 \sim 1$ m/s, $a = 0.25$ mm, $\rho_0 = 1.18$ kg/m³): $F \approx 1.2 \times 10^{-7}$ N, giving Lighthill's regime parameter $\rho_0 F / \mu^2 \approx 430$. That's squarely in the **Fig. 84(b) territory** — wide-angle jet, not the low-forcing symmetric stokeslet (Fig. 82) and not the ultra-narrow high-forcing jet (Fig. 84c).

**Consistency check.** Lighthill's tube result (eq. 209) gives streaming velocity $\bar{u} \sim U_0^2/(4c_0)$. Eq. 209 itself is derived for a tube and doesn't apply to your wall geometry (your advisor caught this at the May 20 meeting), but the *scaling* $\bar{u} \sim U_0^2/c_0$ is robust. With $U_0 \sim 1$ m/s this predicts $\bar{u} \sim 0.7$ mm/s. Your measured `u_n_mean` values are the right order of magnitude — reassuring.

### Which velocity to use for U₀

Your code produces three candidate estimates of the oscillatory velocity:
- `amax` — from the standard-deviation field. Contaminated by turbulence.
- `amax_fit` — Gaussian fit to the STD field. Still contains turbulence.
- **`amax_phase` — RMS across the 8 phase averages.** This isolates the *coherent* oscillatory component.

**`a_phase_array` is the physically correct choice for U₀** — it's the only one that isolates the acoustically-driven oscillation from turbulence, which is what all the theory above assumes.

---

## 3. Literature — Where Your Work Sits

The literature search reframed the paper meaningfully. Your geometry is much closer to **synthetic jet actuators** — zero-net-mass-flux jets driven by an oscillating diaphragm over an orifice — than to classical acoustic streaming. Synthetic jets are usually driven piezoelectrically, but the fluid dynamics is identical to what a wall-mounted speaker produces. This is the community that would recognize your work.

### The essential references

**Riley, N. (2001). "Steady Streaming." *Annual Review of Fluid Mechanics*, 33, 43–65.**
Modern canonical review of steady streaming. Updates Lighthill's framework and covers boundary-layer streaming in detail. First stop for a theory citation beyond Lighthill.

**Glezer, A. and Amitay, M. (2002). "Synthetic Jets." *Annual Review of Fluid Mechanics*, 34, 503–529.**
The defining review of synthetic jets. *This is your geometry.* The paper reviews the vortex-pair formation mechanism at the orifice, the near-field vs far-field structure, and applications in flow control. If you cite one thing to frame your paper, it's this.

**Holman, R., Utturkar, Y., Mittal, R., Smith, B.L., and Cattafesta, L. (2005). "Formation Criterion for Synthetic Jets." *AIAA Journal*, 43(10), 2110–2116.**
The most **directly actionable** result in the whole literature. Defines when a synthetic jet actually forms (coherent vortex rings escape) vs when the orifice just oscillates back and forth. The criterion for axisymmetric jets:
$$\frac{\text{Re}}{S^2} > 0.16$$
where Re is the Reynolds number of the ejection stroke and $S = \sqrt{\omega d^2/\nu}$ is the Stokes number. **Both are computable from your $U_0$ and $f$ for every run in your sweep.** This lets you predict *which of your runs form coherent jets vs which just oscillate*, and compare against what you measure.

**Smith, B.L. and Glezer, A. (1998). "The formation and evolution of synthetic jets." *Physics of Fluids*, 10(9), 2281–2297.**
Foundational experimental paper on synthetic jet formation. Establishes stroke length and formation frequency as the governing scales.

### Direct-geometry references

**Lebrun et al. (1980). "Experimental study of acoustic streaming in the vicinity of orifices."** *Soviet Physics Acoustics* 26, 331. Acoustic streaming around perforated plates — old but geometrically close.

**Henn et al. (2008). "Acoustic PIV: Measurement of the acoustic particle velocity using synchronized PIV-technique."** Uses PIV on sound through a single hole in a wall. Key observation: at high amplitudes vortex generation appears; at low amplitudes only through-flow. **This is exactly the regime transition your frequency sweep should map.**

**Moudjed et al. "Near-field acoustic streaming jet."** Recent work on the near-field of an acoustic streaming jet with PIV-CFD comparison.

### PIV methodology

**Nabavi, Siddiqui, Dargahi (2007).** Synchronized PIV specifically for separating acoustic and streaming velocities. Directly relevant to your phase-interlaced approach.

**Daru et al. (2017). "Acoustic Rayleigh streaming: Comprehensive analysis of source terms..."** Combined LDV + PIV + simulation. Discusses "mutation of streaming at high level" — the regime transitions.

### Classical foundations (for the theory section)

- Lighthill (1978) *Waves in Fluids*, Ch. 4 — your primary theory reference.
- Nyborg (1958) *JASA* 30, 329 — foundational boundary-layer streaming paper.
- Rayleigh (1884) *Phil. Trans. R. Soc.* 175, 1 — the original.
- Westervelt (1953) *JASA* 25, 60 — early analytical work.

### Flow control context (for motivation)

- Cattafesta & Sheplak (2011) *Annu. Rev. Fluid Mech.* 43, 247 — flow control actuators review.
- Jabbal & Zhong (2010) *Phys. Fluids* 22(6) — synthetic jets in a boundary layer, directly parallels the airfoil application context.

### What the literature reframing implies

Your paper is really about **an acoustically-driven synthetic jet at a wall orifice, measured by a novel phase-interlaced PIV technique.** Framing it that way places it in an active flow-control conversation rather than in a somewhat sleepy classical acoustics topic. The Lighthill framework still gives you the theory, but the synthetic jet literature gives you the modern regime map (Holman's criterion) and the modern application context.

---

## 4. What We Covered Last Meeting (May 20)

**You brought in:**
- The bulk-vs-boundary-layer attenuation comparison (the $10^7$ ratio).
- The argument that streaming force is concentrated at the orifice.
- Application of Lighthill eq. 209 to estimate streaming velocity magnitude.

**Advisor's responses:**
1. **Agreed the attenuation argument was correct** and important.
2. **Caught that eq. 209 doesn't strictly apply** — it's derived for a tube boundary layer, not a wall. Its value is as an order-of-magnitude analogy and for the $\bar{u} \sim U_0^2/c_0$ scaling; don't cite it as directly applicable.
3. **The redirect:** *"Can we approximate this as a single concentrated force F and approximate a value of F?"* — this is now the framing.
4. **Follow-up questions raised:**
   - Compute the acoustic power for every run.
   - Check β (eq. 189) explicitly for your frequencies.
   - Independently check whether the acoustic field decays as $1/r^2$ (as expected for a wall-embedded source), separately from the streaming analysis.
   - Consider a simple RANS run with the concentrated force to see if the mean field reproduces.

**Action items you left with:**
1. Compute $P$, $\beta$, $F$, and $\rho_0 F/\mu^2$ for every run using `a_phase_array`.
2. Verify the hole diameter against advisor's slide deck (currently using $d \approx 0.5$ mm).
3. Do a broader literature search beyond Lighthill.
4. Write the methods section once advisor sends the LaTeX document.

**Next meeting was set for ~2 weeks out. Advisor was to create the LaTeX document. You were to message him if you didn't hear back.** (Given the summer gap, it's unclear whether that document ever got created — worth asking today.)

---

## 5. Quick Summary

- Paper: PIV of a wall-mounted speaker in a wind tunnel, using an 8-phase 45°-lag stroboscopic technique that separates streaming from oscillation.
- Your role: acoustic streaming theory and analysis section, plus writing methods once the LaTeX doc exists.
- Key physics: only attenuated waves stream; boundary-layer attenuation in the orifice dominates bulk attenuation by ~10⁷; force is concentrated at the orifice; treat it as a point force $F = \frac{1}{2}\rho_0 U_0^2 A_{\text{hole}}$.
- Regime: $\rho_0 F/\mu^2 \sim 430$ — Lighthill Fig. 84(b), wide-angle jet.
- Big literature reframe: your geometry is really a **synthetic jet**. Holman et al. 2005 formation criterion $\text{Re}/S^2 > 0.16$ is directly computable per run and is your best modern regime map.
- Where you left off: advisor redirected toward the concentrated-force framing; you were to compute P, F, and the regime parameter for every run and hunt for more literature.

---

## 6. Talking Points for This Meeting

Structured for a natural flow: casual catch-up → project state → next steps.

### Catching up
- Summer wrapped, back to school, senior year now.
- Ask how his summer was, whether anyone else was in the lab.
- Check status: did he end up creating the LaTeX methods document over the summer? If yes, get access. If no, no worries, but flag it as a near-term thing.

### Where we left off
- "Last time we talked, the plan was to treat the speaker as a concentrated force at the orifice and compute F from acoustic power. I've been away from the analysis over the summer so I want to make sure I'm still on the right track before I dig back into the code."
- Reconfirm: $F = \frac{1}{2}\rho_0 U_0^2 A_{\text{hole}}$, using `a_phase_array` for $U_0$.
- Confirm the orifice diameter (you have $\sim 0.5$ mm noted from his slides but it's worth verifying).
- Confirm that the regime parameter $\rho_0 F/\mu^2 \sim 430$ target puts us in Fig. 84(b) — narrow-ish jet — and that this matches what he's seen in the data.

### New from literature
- **This is the pitch.** "I did a broader literature search and I think our geometry is much closer to a synthetic jet than to classical acoustic streaming in the Lighthill sense. Glezer & Amitay 2002 and Holman et al. 2005 are the canonical references."
- The Holman criterion $\text{Re}/S^2 > 0.16$ is directly computable per run and predicts *which of our runs form coherent jets vs just oscillate*. That's a sharper regime map than Lighthill's $\rho_0 F/\mu^2$.
- Question for him: does he think we should reframe the paper explicitly as a synthetic jet paper, or keep the classical-streaming framing and just reference the synthetic jet literature?

### Next steps to propose
1. **Compute the full table:** for every run, tabulate $f$, $U_0$ from `a_phase_array`, $P$, $\beta_{\text{bulk}}$, $F$, $\rho_0 F/\mu^2$, $\text{Re}$, $S$, and $\text{Re}/S^2$.
2. **Frequency-scaling plot** of streaming velocity vs $f$ — the slope discriminates between mechanisms (bulk streaming $\propto f^2$, boundary layer $\sim$ flat, Rayleigh $\propto 1/f$).
3. **Fit Schlichting's laminar-jet centerline profile** $\bar{u}(z, 0) = 3F/(8\pi\mu z)$ to `u_n_mean` to get $F$ independently from the measured jet. The ratio $F_{\text{measured}}/F_{\text{acoustic}}$ is a **conversion efficiency** — a reportable number.
4. **Check acoustic field decay** ($1/r$ vs $1/r^2$) as he asked — separately from the streaming analysis.
5. **Compute Reynolds stress divergence** from the 8 phases — this is the analysis uniquely enabled by his interlaced technique and probably deserves its own subsection.
6. Ask about **methods section timing** and access to the LaTeX doc.
7. **Fall project check-in:** wind-on phase-locked PIV plan. Who's in the lab? Who's running with you?

### One or two things to flag from the code review
- The `momentum_integral` assumes axisymmetry, which is questionable given airfoil surface curvature — worth flagging as an assumption we're documenting.
- `f_array` has some duplicated frequencies (505.8, 239.77, 197.77 each twice) — check whether those are intentional repeats or an error.
- The signal-to-noise threshold in `magnitude_identification` might silently zero out weak low-frequency runs (especially 29.75 Hz) — worth a warning message.

---

*End of recap.*
