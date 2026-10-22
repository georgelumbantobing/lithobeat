# A constrained inverse-medium theorem

This is a finite-dimensional convex result, proved below. It does not assert optimal subjective sound or globally optimize arbitrary nonlinear learned speakers.

## Definitions and assumptions

The real waveform is a vector of a fixed finite length. The complex two-dimensional analysis image indexes frequency and frame. Analysis is a fixed real-linear map into complex arrays, with their real Euclidean inner product. Its real adjoint takes the real part of inverse FFT frames and overlap-adds them.

The implementation uses full, orthonormal FFTs, a sine (square-root periodic Hann) window and half-window hop. Length is padded to a hop multiple, with circular frame indexing. The squared overlapping windows sum to one. Consequently analysis preserves squared norm and analysis followed by its adjoint reproduces the waveform exactly. This also specifies the boundary convention completely; it is not an ordinary causal loudspeaker model.

Every playback profile is a fixed finite complex multiplier for each frequency. It may be measured or estimated **before** optimization, but is held fixed during the solve. Media mixture weights are nonnegative and sum to one. The target image is fixed, independently of the candidate waveform. The positive ridge weight penalizes deviation from the source. The positive amplitude cap bounds every sample, including the padding.

The objective is half the weighted sum, over media, of squared Euclidean distances between each profile times the candidate STFT and the target, plus half the ridge weight times the squared waveform change. The feasible set is the closed amplitude box.

The code in `Problem.value_gradient` is the executable definition. The objective compares **complex coefficients**, so it penalizes phase and magnitude errors. It is not a log-spectrum, perceptual loudness, or human-preference objective. Optimizing arbitrary coefficient pixels and subsequently synthesizing would generally break STFT consistency. Here every evaluated image is the analysis of a feasible waveform.

## Theorem: existence, uniqueness, global optimality, convergence

Under the assumptions above:

1. A unique global minimizer exists on the amplitude box.
2. A feasible candidate is optimal exactly when the real inner product of its gradient with every feasible displacement is nonnegative. Equivalently, it equals its own projected-gradient update for any positive step.
3. With the reciprocal of the implemented Lipschitz bound as step size, projected gradient converges to that unique minimizer. Every iterate respects the amplitude constraint, and objective values are nonincreasing.
4. `gap_bound` bounds the objective suboptimality from above in exact arithmetic.

**Proof.** Each medium's loss is a squared norm of an affine real-linear function and therefore convex. The ridge term has Hessian equal to the positive ridge weight times the identity. The total objective is thus strongly convex with that weight. It is continuous, and the nonempty compact feasible box gives existence. Strict convexity on the convex feasible set gives uniqueness. The differentiable convex first-order variational inequality is necessary and sufficient for global optimality, and the Euclidean projection characterization gives the fixed-point statement.

The gradient is the real adjoint of analysis applied to the sum of conjugate-profile-weighted residuals, plus the ridge displacement. Its Hessian is analysis-adjoint, then multiplication by the mixture of squared profile magnitudes, then analysis, plus the ridge identity. Because analysis is an isometry, the largest mixture power plus the ridge weight is an upper bound on the Hessian norm. This is `lipschitz` in code. All Hessian eigenvalues lie between the ridge weight and this bound. The affine gradient step therefore contracts distances by at most one minus the ridge weight divided by that bound; projection onto a convex box is nonexpansive, so the composite map contracts and converges to its unique fixed point. The descent lemma and projection inequality imply an objective decrease of at least half the Lipschitz bound times the squared step length.

For the certificate, strong convexity supplies a lower quadratic support function at the current feasible candidate: objective there, plus the gradient-displacement inner product, plus half the ridge weight times squared displacement. Minimize that quadratic support over the box. Completing the square shows that its minimizer is the candidate minus gradient divided by ridge, projected onto the box. Subtract its support value from the current objective. The resulting nonnegative number is exactly `gap_bound`. The true minimum is at least the minimized support, proving the claimed upper bound on suboptimality. QED.

Floating point arithmetic is not interval arithmetic: the printed certificates are numerical evaluations of a proven exact-arithmetic bound, with roundoff limitations. Tests check adjointness, gradients, a closed-form case, descent, and an independent SciPy constrained solver. These numerical checks supplement the proof rather than proving it.

## Learned-medium scope and stability

A fixed learned complex profile is permitted by the theorem regardless of how it was learned. That guarantees optimization of its predictions, not their physical accuracy. `fit_diagonal` implements ridge least squares on aligned paired STFT coefficients. Its synthetic held-out test demonstrates the interface, not acoustic generalization.

There is also a general model-transfer statement that needs no convexity: suppose the true and predicted objectives differ by at most a common bound uniformly over the entire feasible box. A predicted optimizer with objective gap at most a specified tolerance has true-objective regret at most twice that uniform discrepancy bound plus the optimization tolerance. Proof: bound the true objective at the computed candidate by its predicted objective plus discrepancy; apply approximate predicted optimality against a true optimizer; bound that optimizer's predicted objective by its true objective plus discrepancy. Existence of the true optimizer must be assumed (for example, continuity on the compact box). This is conditional; this package does not establish a physical uniform discrepancy bound.

An arbitrary nonlinear learned forward system can yield a nonconvex objective with local minima. Replacing the fixed linear map by that system invalidates the strong-convexity and convergence argument unless new assumptions establish them. A nonnegative weighted bank of fixed linear models remains within the theorem; `balanced` demonstrates the expected-loss version, not worst-case optimization.

## Headroom and boundaries

The proven constraint is sample amplitude of the optimized digital waveform, with no post-solve normalization or clipping. It is not a true-peak reconstruction bound, an acoustic sound-pressure constraint, or protection against clipping in a downstream device. A 4x polyphase oversampling peak is reported only as a diagnostic. Float64 WAV export is read back and checked for exact equality. Players unable to read this format can convert locally, then recheck peaks; format conversion is outside the certificate.

All padded samples are optimization variables and are exported. Periodic indexing can couple the end and beginning; real production deployment needs guard regions, explicit boundary models, stereo/image constraints and streaming treatment. A fixed linear FIR forward operator with its correct adjoint can replace the diagonal surrogate without losing convexity; its operator norm must replace the current Lipschitz bound. It has not been implemented here.
