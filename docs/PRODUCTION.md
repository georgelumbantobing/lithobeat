# Path to actual production audio

Use recordings you own or have an explicit license to process and redistribute. Keep unreleased production recordings out of public fixtures unless authorized. The included beat is algorithmically generated from oscillators and seeded noise, with no sampled commercial recording.

For each actual playback device and room, record broadband calibration material at fixed volume, position and device settings; preserve microphone calibration and gain. Align reference and response, account for latency, and split training, validation and testing by recording/session. Fit a fixed model on training only and evaluate prediction error on withheld material. Current `fit_diagonal` consumes aligned complex STFT arrays; there is no turnkey microphone acquisition or autoalignment pipeline here.

A diagonal fit is only a starting approximation. Prefer a measured causal FIR/linear operator when cross-frame effects matter, and explicitly characterize nonlinear distortion and level dependence. Do not describe a synthetic earbud response as an AirPods measurement. Substitute a validated forward operator and its tested adjoint before claiming real hardware performance. A frozen bank of linear operators can represent multiple measured conditions.

Prespecify the target, headroom, preservation weight and evaluation metrics before listening comparisons. Evaluate unfamiliar held-out tracks and rooms. Report negative results, cross-device costs, spectral error and processing change alongside properly measured loudness, true peak, stereo image and distortion. Use blind loudness-matched listening for subjective claims. Low coefficient error alone does not establish preference, loudness, groove recovery or audibility.

The 16 kHz, four-second mono demonstration is deliberately small and reproducible; it is not a full-band mastering product. Expand to 44.1/48 kHz and stereo with boundary and linking constraints, maintain sample/true-peak checks after encoding, and validate memory/runtime and acoustic prediction independently. No real production-audio claim is made by this release.
