# MNIST architecture experiments — 2026-08-30

## Baseline: 2-layer (784→128→10)
- 5 epochs, full 60k dataset: 97.29% acc, final loss 0.0714, time: ??? 
- 50 epochs, 1000-image subset: 86.99% acc, final loss 0.0983 — overfitting ceiling demo
- 25 epochs, full 60k dataset: 98% acc, finall loss 0.0184, time: 111.73s

## 3-layer (784→128→64→10)
- 25 epochs, full 60k dataset: 98.13% acc, loss →0.0024, time: 114.37s

## Observations
- More data (60k vs 1k) beat more epochs (5 vs 50) — dataset size >> training duration for generalization
- Extra depth (3-layer vs 2-layer) didn't meaningfully beat 2-layer on this task — MNIST wasn't capacity-starved

## Activation function comparison (784→128→10, full 60k dataset)
 
- 15 epochs: relu 97.89% / leaky_relu 97.69% / prelu 95.10% / elu 97.52% — all ~59s
- 50 epochs: relu 97.92% / leaky_relu 98.04% / prelu 98.18% / elu 97.91% — all ~193s
## Observations
 
- At 15 epochs, PReLU trailed badly (95.10%, a real 2-3pt gap) — its extra learnable
  parameter (negative slope α) hadn't converged yet.
- At 50 epochs, PReLU flipped to the best performer (98.18%). relu/leaky_relu/elu were
  already near their ceiling at 15 epochs and barely moved by 50.
- Takeaway: more expressive activations (learnable params) aren't automatically better —
  they need enough training budget to actually use the extra flexibility. Given equal
  budget here, PReLU edged out the fixed-shape alternatives.
- Training time was ~identical across all four activations on CPU — the exponential in
  ELU didn't show up as a measurable cost here. Worth remembering this doesn't transfer
  to hardware: on CPU the matmuls dominate regardless of activation, but in silicon an
  exponential unit is real extra area/power that a comparator (ReLU) doesn't need.
- For the thesis specifically: plain ReLU is still the practical choice — cheapest in
  hardware (comparator + mux, no multiply, no learnable param to train correctly) and
  competitive in accuracy even though it wasn't the single best number here.
## Parked ideas (not in current thesis scope, revisit if time allows)
 
- **Sparse/masked connectivity**: simulate non-fully-connected layers in software
  (weight mask, zero out a fraction of connections) to see how much accuracy is lost
  vs. how much wiring/area could be saved. Directly relevant to the near-memory
  architecture's stated area-scaling weakness.
- **Pruning**: train dense, then remove low-magnitude weights post-hoc (optionally
  fine-tune after). Real technique, could quantify how much of the 128→32 layer's 4096
  connections are actually load-bearing. Same motivation as above — could become a
  "future directions" writeup or, if compelling enough, a discussion with Jurgo about
  folding into scope.
 
## Precision ladder: fp32 -> binary (784->128->10 baseline, fc1 layer only, post-training quantization)
 
Baseline model: 25-epoch, full 60k dataset, 97.81% fp32 accuracy (trained_model.pt).
All quantization applied to fc1 weights only, post-training (no retraining/fine-tuning
after quantizing). Same neuron indices visualized at every precision level for direct
comparison.
 
| precision      | accuracy | method |
|----------------|----------|--------|
| fp32 (baseline)| 97.81%   | native |
| fp16           | 97.81%   | `.half()` |
| bf16           | 97.82%   | `.bfloat16()` |
| int8           | 97.82%   | affine, scale = max(\|w\|)/127 |
| int4           | 97.67%   | affine, scale = max(\|w\|)/7 |
| ternary (t=1.0)| 85.64%   | threshold = 1.0 x mean(\|w\|), sign() |
| ternary (t=0.7)| 75.08%   | threshold = 0.7 x mean(\|w\|), sign() |
| binary         | 77.50%   | sign(w), scale = mean(\|w\|) |
 
## Observations
 
- **fp32 -> int8 is nearly free.** Accuracy differences across fp32/fp16/bf16/int8 are
  within 0.01-0.02%, statistical noise. 256 discrete levels (int8) is still more than
  enough resolution for this network's weight distribution — the network never needed
  fp32's precision in the first place.
- **int4 is the first real crack.** -0.14pt drop, small in absolute terms but a
  qualitatively different pattern from the fp32-int8 plateau. Visually, degradation is
  uneven across neurons: some (weights concentrated in few dominant values) survive int4
  cleanly, others (weights spread diffusely across many similar small magnitudes)
  collapse to visible speckle. Only 16 levels starts to matter, and it matters more for
  some neurons' weight distributions than others.
- **Ternary is a cliff, not a slope.** 85.64% (t=1.0) is a 12+ point drop from int4 —
  qualitatively different failure mode, not a continuation of the int4 trend. Visually,
  every tested neuron's receptive field collapses to unstructured speckle — the smooth,
  recognizable spatial patterns visible through int4 are gone entirely at ternary.
  Confirms the literature's framing (BitNet etc.): naive post-training ternary
  quantization degrades hard; networks need to be trained *aware* of the ternary
  constraint (quantization-aware training) rather than quantized after the fact.
- **Lower threshold made ternary WORSE, not better (75.08% vs 85.64%) — counterintuitive,
  worth remembering.** Hypothesis going in was "more permissive threshold -> more
  surviving weights -> more information -> better accuracy." Wrong: a stricter threshold
  (1.0x mean) only let genuinely high-magnitude, likely-important weights survive as
  ±1; the permissive threshold (0.7x mean) let marginal/borderline weights through too,
  adding noise rather than useful signal. More non-zero weights is not automatically
  better if the extra ones are low-confidence. Same "more flexibility isn't free" pattern
  as the PReLU result, in a different form.
- **Binary (77.50%) lands between the two ternary variants**, better than ternary(0.7),
  worse than ternary(1.0). No zero option at all — every connection stays "on" at full
  forced magnitude, no ability to prune weak connections to silence the way ternary can.
  Visually the most saturated/blocky of all — no near-white regions anywhere, unlike
  ternary which had visible white space where weights zeroed out.
- **Net takeaway for the near-memory design**: post-training quantization holds up fine
  down through int4-ish precision, but ternary specifically (the thesis's actual target)
  needs quantization-aware training to be viable — this data is a concrete argument for
  why QAT (or at minimum fine-tuning after quantizing) belongs in the methodology, not
  just "train fp32, then round."
## Parked ideas (not in current thesis scope, revisit if time allows)
 
- **Sparse/masked connectivity**: simulate non-fully-connected layers in software
  (weight mask, zero out a fraction of connections) to see how much accuracy is lost
  vs. how much wiring/area could be saved. Directly relevant to the near-memory
  architecture's stated area-scaling weakness.
- **Pruning**: train dense, then remove low-magnitude weights post-hoc (optionally
  fine-tune after). Real technique, could quantify how much of the 128->32 layer's 4096
  connections are actually load-bearing. Same motivation as above — could become a
  "future directions" writeup or, if compelling enough, a discussion with Jurgo about
  folding into scope.
- **Quantization-aware training (QAT)**: given how badly naive post-training ternary
  degrades (see above), the natural next experiment is training WITH the ternary
  constraint applied during the forward pass from the start, rather than quantizing
  a fp32-trained model after the fact. Directly tests whether QAT closes the ~12-22pt
  gap seen here.
 
## Quantization-aware training (QAT) — straight-through estimator, ternary fc1/fc2
 
Followed up on the naive post-training ternary result (85.64%, see precision ladder
section above) by actually training WITH the ternary constraint active, instead of
quantizing a finished fp32 model after the fact.
 
### Method: straight-through estimator (STE)
 
`sign()`/`round()` are step functions — flat almost everywhere, so their true gradient
is 0 almost everywhere. Backpropagating through them honestly would freeze every weight
permanently (`w -= lr * 0 = w`, no update ever happens). STE fixes this: forward pass
computes the real quantized value, but the gradient path is deliberately severed
(`.detach()`) so that during backward, the quantization op is treated as identity
(gradient = 1) instead of its true (zero) derivative. Not mathematically honest, but
gives the optimizer a workable, non-zero signal.
 
```python
def ternary_ste(w, threshold_multiplier=1.0):
    threshold = threshold_multiplier * w.abs().mean()
    mask = (w.abs() >= threshold).float()
    q = torch.sign(w) * mask
    q_scaled = q * threshold
    return w + (q_scaled - w).detach()
```
 
Applied inside `forward()` via `F.linear(x, ternary_ste(self.fc1.weight), self.fc1.bias)`
instead of calling `self.fc1(x)` directly — this quantizes the weight fresh every
forward pass while training a full-precision "shadow" copy underneath via normal
autograd.
 
### Critical distinction: "live" accuracy vs. "frozen" accuracy
 
These are NOT the same number, and the gap between them is the single most important
finding of this session.
 
- **Live accuracy**: measured during/after training, using `ternary_ste()` applied
  fresh every forward pass. The threshold is recomputed from whatever the current
  (still-shifting, still fp32) shadow weights happen to be at that moment.
- **Frozen accuracy**: after training, weights are quantized ONE FINAL TIME and
  permanently overwritten (`model.fc1.weight.data = ternary_ste(...)`), simulating
  what real hardware actually has — fixed values, no live re-quantization possible.
Frozen accuracy is the number that actually matters for the thesis — it's the only one
representative of what physical ternary weights in silicon could achieve. Live accuracy
is informative about training dynamics but NOT a valid stand-in for hardware performance.
 
### Results
 
| config | live acc | frozen acc |
|---|---|---|
| fc1-only ternary, fp32 fc2 (single run) | ~97.1-97.4% | 92.98% |
| fc1+fc2 ternary, full (run A) | ~96.9% | 88.14% |
| fc1+fc2 ternary, full (run B, different init) | ~96.9% | 82.49% |
| fc1+fc2 ternary, 3-seed sweep (seeds 0/1/2) | 96.56-96.85% | 72.68% / 76.75% / 79.94% |
 
3-seed sweep: mean frozen 76.46%, spread (max-min) 7.26 points, live accuracy stable
within ~0.3pt across seeds.
 
### Observations
 
- **QAT beats naive post-training quantization decisively.** Even the worst frozen QAT
  result (72.68%) beats naive (85.64%)... actually check this — naive (85.64%) beats
  the WORST QAT seed (72.68%) but QAT's best runs (88-93%) beat naive comfortably.
  QAT's outcome is not uniformly better than naive — it depends heavily on seed luck
  for the full-ternary case. Worth re-stating precisely: partial (fc1-only) QAT reliably
  beat naive; full QAT's advantage over naive is real on average across seeds but not
  guaranteed on any single run.
- **Live accuracy is stable across seeds (~96.5-97.4%, tight spread); frozen accuracy
  is highly unstable (7.26 point spread across just 3 seeds).** The instability lives
  entirely in the freeze step, not in training itself. This means a single QAT run's
  reported number is not trustworthy without multiple seeds — a real methodological
  requirement for any future thesis-relevant QAT experiments (report mean + spread,
  not a single number).
- **Partial/mixed-precision quantization outperformed full quantization.** fc1-only
  ternary (92.98% frozen) beat fc1+fc2 full ternary (72.68-88.14% frozen) in every
  comparison. Directly suggests a mixed-precision design — ternary only where the
  near-memory area savings actually matter most (Layer 1, ~4096 of ~4128 total
  weights in the real thesis architecture) while keeping smaller layers (32->8,
  8->1) at higher precision — could be a stronger real design than uniform
  end-to-end ternary. Worth raising with Jurgo as a possible scope refinement,
  not just future work — this is a small, well-motivated change to "which layers
  are ternary," not a departure from the near-memory ternary concept itself.
- **Caught and corrected a false positive along the way**: naive post-hoc fine-tuning
  (start from ternary-quantized weights, then continue training normally with no STE)
  appeared to "recover" accuracy to ~97.6%, but `torch.unique()` on the resulting
  weights showed ~90k distinct values — the weights had silently drifted back to full
  fp32 precision, not stayed ternary. That result was invalid; STE-based QAT (which
  keeps weights genuinely ternary on every forward pass throughout training) is the
  correct method, and produces the real 72-93% frozen range above instead of a
  fake ~97%.
### Open questions / next steps
 
- Why is frozen accuracy so seed-sensitive specifically at freeze time, when live
  accuracy isn't? Possible hypothesis: some seeds land the shadow weights in a
  configuration where the final live threshold/mask assignment is "fragile" (many
  weights sitting right at the threshold boundary, so freezing tips more of them
  the wrong way) vs. other seeds landing somewhere more robust. Not yet tested.
- Try the mixed-precision idea properly: fc1 ternary + fc2 int8 (rather than fc2 full
  fp32) as a middle ground — does fc2 need FULL precision to stabilize things, or
  would a cheaper-than-fp32-but-more-than-ternary fc2 achieve similar frozen accuracy
  to the fc1-only result?
- Average frozen accuracy over more seeds (5-10) once there's time, to get a more
  reliable mean +/- spread rather than just 3 samples.
- This entire investigation used a fixed threshold rule (mean(|w|) x multiplier).
  Not yet explored: learned/adaptive thresholds during QAT (rather than a fixed
  formula), which is closer to how real published ternary-network methods (e.g.
  BitNet-style approaches) typically operate.
 
## Binary task test: Wisconsin Breast Cancer (WDBC) — matches thesis output shape
 
Motivation: MNIST is 10-class with softmax output, but the thesis architecture is
genuinely binary (single sigmoid output neuron), so MNIST results don't directly
validate the actual output shape being used. Switched to sklearn's built-in WDBC
dataset: 569 samples, 30 real-valued features, genuinely binary (malignant=0,
benign=1), class split 212/357 (62.28% majority-class floor). Architecture:
30->16->8->1, sigmoid output, BCELoss — first real use of the binary setup derived
by hand on day one, at network scale.
 
Data prep notes: 80/20 train/test split (455/114 samples). StandardScaler fit on
train only, applied to test (avoid leakage) — feature scales vary hugely across the
30 raw measurements (e.g. "mean radius" ~6-28 vs "mean area" ~150-2500), unlike
MNIST's pre-normalized 0-1 pixel values, so scaling was necessary here in a way it
wasn't for MNIST.
 
### Results
 
| config | test accuracy |
|---|---|
| "always guess majority class" floor | 62.28% |
| fp32 baseline | 96.49% (train 96.48%, no overfit gap) |
| naive ternary (post-training) | 76.32% |
| naive binary (post-training) | 88.60% |
| QAT ternary, 5-seed mean (frozen) | 62.28% (spread: 0.00pt across all 5 seeds) |
| QAT binary, 5-seed mean (frozen) | 95.09% (spread: 5.26pt) |
 
### Observations
 
- **QAT ternary collapsed to EXACTLY the majority-class floor, on every single seed,
  zero variance.** This is not "ternary is bad" in the usual degraded-accuracy sense
  seen with MNIST — it's a distinct, fully diagnosed failure mode (see below).
- **QAT binary, by contrast, worked excellently** — 95.09% mean, beat naive binary
  by ~6.5pts, close to fp32. Binary has no zero bucket, so it structurally cannot
  suffer the same collapse ternary just hit (every weight forced to be meaningfully
  +-something, never exactly zero).
- **Diagnosed root cause of the ternary collapse**: checked per-layer zero fraction
  after freezing (seed 0) — fc1 52.1% zero, fc2 56.2% zero, fc3 50.0% zero (of 8
  total weights). No single layer fully collapsed to 100% zero, but roughly half of
  every layer's weights vanished. Checked raw sigmoid outputs on the full test set:
  ALL 114 outputs landed in a narrow 0.589-0.628 window — the network produces
  essentially the same output regardless of input, i.e. it isn't discriminating
  between samples at all. Root cause: fc3's bias was 0.4324, a comparatively large,
  UNQUANTIZED (biases were never touched by any quantization function this whole
  project) fp32 value. With ~50%+ of every layer's weights zeroed, the actual
  input-dependent signal reaching the final layer is small and noisy; the untouched,
  comparatively large bias dominates the sum, and sigmoid(mostly-bias + tiny noise)
  lands above 0.5 for virtually every input regardless of the actual sample.
- **This is a genuinely different failure mode from anything seen in MNIST** — MNIST's
  much larger layers (128, 32 width vs. this network's 16, 8) apparently had enough
  surviving non-zero weights that bias-domination never fully collapsed the output.
  Smaller/narrower networks appear more vulnerable to this specific failure —
  directly relevant to the thesis's own 32->8->1 tail (Layers 2/3 are comparably
  narrow to this experiment's fc2/fc3).
- **Not yet tested**: quantizing the bias terms too (same threshold/sign logic as
  weights). Strong hypothesis this would prevent the collapse, since the bias
  couldn't silently retain outsized, untouched influence relative to the quantized
  weights feeding into the same sum. Worth testing directly as a fast, high-value
  follow-up before assuming ternary is unworkable on narrow layers.
- **Practical implication for the actual thesis architecture**: Layer 3 in particular
  (8->1, single output neuron) is exactly the kind of narrow layer that produced this
  failure here. Worth treating "does the narrowest layer(s) need special handling
  (bias quantization, different threshold, or simply staying higher-precision) to
  avoid bias-domination collapse" as a concrete open question for the real design,
  not just a MNIST/toy-dataset curiosity.
### Open questions / next steps
 
- Test bias quantization directly (quantize fc1/fc2/fc3 biases the same way as their
  weights) — does this prevent the ternary collapse seen here?
- Does the same collapse reproduce on the 3-layer MNIST network's narrow tail, or is
  it specific to this dataset/architecture width? (MNIST's smallest layer, fc3 in the
  3-layer experiment, was 64->10 — still much wider than this experiment's 8->1.)
  Worth a targeted comparison rather than assuming.
- QAT binary's 5.26pt spread, while much better than ternary's collapse, is still a
  real spread worth investigating with more seeds — is 5 seeds enough to trust the
  95.09% mean, or does it need 10+ like a proper study would use?
 
