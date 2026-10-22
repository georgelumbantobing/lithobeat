# A 35-second reproducible demonstration

Use `examples/generated/tiktok_portrait.png` as the portrait visual and `overview.png` / `spectrograms.png` for detail shots. No separate presentation is needed.

- 0–6 s: “What if you mastered for what a speaker actually reproduces?” Show the original generated beat and two invented response curves.
- 6–15 s: “I treat the beat as a time-frequency image and send it through a simulated playback model.” Show the spectrogram and the source-through-earbud simulation.
- 15–25 s: “Then I optimize the input, while every sample stays inside the headroom limit.” Show the earbud master through the same simulation and the convergence/peak plots.
- 25–35 s: “This convex model has a unique global optimum. The code, proof and synthetic audio are reproducible. Real speaker tests come next.” Show the cross-device error chart and repository link when published.

Persistent on-screen label: “Synthetic playback models. No AirPods or club measurements.” Error is a mathematical proxy, not listening preference. Raw before/after audio has different RMS levels; do not present it as a loudness-matched preference test. Use a common playback gain for the raw scientific A/B, or produce a separately labeled loudness-matched listening version. Do not imply that matching loudness preserves the optimized solution.

Source WAVs are digital masters. Files named `through_*` are outputs of the virtual medium renderer, not recordings of a speaker. Downstream simulated output can exceed the master cap; it is neither constrained nor presented as a distributable master.
