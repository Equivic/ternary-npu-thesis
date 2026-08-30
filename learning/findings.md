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
 
