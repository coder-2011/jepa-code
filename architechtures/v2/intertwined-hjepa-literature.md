# Intertwined H-JEPA Literature Map

## Purpose

This note maps the literature that should inform Intertwined H-JEPA.

This note includes both the current implementation and older design hypotheses. Current implementation notes should be read as the source of truth when they differ from the proposed cross-depth residual-delta variants below.

Current implementation commitments:

1. JEPA-style prediction happens inside the residual stream, not only at the output.
2. The target for layer `l` is produced by an EMA copy of the same layer's compressor, applied to that layer's post-attention states.
3. The JEPA loss predicts the next-token latent delta: `delta_l[:, t] -> target_z_l[:, t+1] - z_l[:, t]`.
4. The predictor is auxiliary-only; the residual stream is updated by a separate transition MLP, not by projecting `z_l + delta_l`.

That puts the design between JEPA, EMA/self-distillation, predictive coding, residual dynamics, and language-model auxiliary losses.

## Source Tiers

### Tier 1: Directly Canonical For This Project

#### I-JEPA

Source: <https://arxiv.org/abs/2301.08243>

Core idea:

- Predict latent target-block representations from a context block.
- Avoid direct pixel/token reconstruction.
- Masking policy matters: large semantic target blocks and informative context blocks are central to representation quality.

Design implication:

- Our target should be a latent state, not token identity.
- The prediction target should be semantically meaningful enough that the predictor cannot solve it through trivial local copying.
- For text, masking and future-state prediction should be treated as architecture-level choices, not just data augmentation.

#### BYOL

Source: <https://arxiv.org/abs/2006.07733>

Core idea:

- Online network predicts a target network representation.
- The target network is updated by a slow moving average of the online network.
- No negative pairs are required.

Design implication:

- EMA teacher mechanics are well precedented.
- The teacher should be stop-gradient.
- We need explicit tests for no target gradients and correct EMA update timing.

#### data2vec

Source: <https://arxiv.org/abs/2202.03555>

Core idea:

- A masked student predicts contextualized latent representations of the full input.
- The method is modality-general across speech, vision, and language.
- It predicts latent contextual targets rather than modality-specific local labels.

Design implication:

- Text latent prediction is not inherently weird; there is precedent.
- The target should probably be contextualized by a teacher path, not simply an embedding lookup.
- A borrowed tokenizer path plus LM head is fine, but the JEPA loss should remain latent-state based.

#### LLM-JEPA

Source: <https://arxiv.org/abs/2509.14252>

Core idea:

- Applies JEPA-style embedding-space objectives to language models.
- Keeps language-model-style generative/reconstruction capability in the training picture.
- Reports gains over standard LLM training objectives across several model families and datasets.

Design implication:

- Adding an LM head is consistent with the direction of language JEPA work.
- We should explicitly ablate JEPA-only, LM-only, and JEPA+LM.
- The LM head should not become the only useful objective; the architectural question is whether internal predictive deltas improve representations and generation.

### Tier 2: Direct JEPA Extensions

#### V-JEPA

Source: <https://arxiv.org/abs/2404.08471>

Core idea:

- Feature prediction can stand alone as a video self-supervised objective.
- V-JEPA trains without reconstruction, negatives, text, pretrained image encoders, or other supervision.

Design implication:

- A pure latent prediction objective is viable at scale.
- The strongest lesson for us is discipline: keep reconstruction/token prediction separate from latent predictive learning and test their interaction.

#### V-JEPA 2

Source: <https://arxiv.org/abs/2506.09985>

Core idea:

- Scales action-free JEPA pretraining on internet video, then aligns with LLMs and action-conditioned world models.
- Shows JEPA-style representations can support prediction and planning, not only classification.

Design implication:

- The "world model" framing is relevant, but only after the small auxiliary next-latent mechanism works.
- Do not overbuild planning/world-model infrastructure in v1.

#### LeJEPA

Source: <https://arxiv.org/abs/2511.08544>

Core idea:

- Argues that JEPA embeddings should follow an isotropic Gaussian distribution.
- Introduces SIGReg to prevent collapse without stop-gradients, EMA teachers, or schedules.

Design implication:

- Collapse diagnostics are mandatory.
- SIGReg is a serious later addition, but v1 should not depend on it before the basic auxiliary next-latent mechanism is proven.
- If EMA plus gating still collapses, SIGReg or a VICReg-style variance/covariance loss becomes a top-priority fix.

#### C-JEPA

Source: <https://arxiv.org/abs/2410.19560>

Core idea:

- Connects JEPA with contrastive/self-supervised regularization.
- Critiques EMA from I-JEPA as insufficient by itself for preventing entire collapse.
- Combines I-JEPA with VICReg-style variance/invariance/covariance regularization.

Design implication:

- EMA teacher alone is not a collapse solution.
- We should log representation variance and effective rank from the first training smoke.
- Add variance floor before more complicated fixes.

#### Rethinking JEPA / SALT

Source: <https://arxiv.org/abs/2509.24317>

Core idea:

- Replaces EMA teacher in video JEPA with a frozen teacher trained separately by pixel reconstruction.
- Argues frozen teachers can decouple optimization and improve compute transparency.

Design implication:

- EMA is a reasonable v1 choice because the user specified it.
- But a frozen-teacher ablation is a strong future baseline if EMA teacher instability appears.

#### EB-JEPA Library

Source: <https://arxiv.org/abs/2602.03604>

Core idea:

- Presents modular energy-based JEPA examples for images, video, and action-conditioned world models.
- Emphasizes the importance of regularization components for preventing collapse.

Design implication:

- Keep our implementation modular: compressor, EMA compressor, predictor, transition MLP, regularizer.
- Avoid hiding the energy/loss contract inside opaque trainer code.

### Tier 3: Collapse Prevention and Siamese/Masked Precedent

#### VICReg

Source: <https://arxiv.org/abs/2105.04906>

Core idea:

- Self-supervised agreement objectives can collapse to constant outputs.
- Explicit variance and covariance regularization can prevent collapse without relying only on architectural asymmetry.

Design implication:

- Add metrics for:
  - per-dimension standard deviation
  - covariance off-diagonal magnitude
  - effective rank
  - prediction/target norm ratio
- Keep a simple variance-floor regularizer ready.

#### Masked Siamese Networks

Source: <https://arxiv.org/abs/2204.07141>

Core idea:

- Match a masked-view representation to an unmasked-view representation.
- Processing only unmasked patches improves scalability in ViTs.

Design implication:

- Our v1 dense sequence path is fine for simplicity.
- Later, sparse target/context computation may be useful, but should not be introduced before correctness tests are stable.

### Tier 4: Predictive Coding and Residual/Error Dynamics

#### PredNet

Source: <https://arxiv.org/abs/1605.08104>

Core idea:

- Hierarchical video prediction model inspired by predictive coding.
- Each layer makes local predictions and forwards deviations from those predictions upward.

Design implication:

- The earlier residual-delta-injection design was closer to predictive coding than ordinary JEPA heads.
- In the current implementation, the auxiliary predictor does not update the residual stream; track whether the separate transition MLP and auxiliary JEPA loss still produce useful latent dynamics.

#### Dynamic Predictive Coding

Source: <https://www.biorxiv.org/content/10.1101/2022.06.23.497415v3.full>

Core idea:

- Hierarchical sequence learning where higher levels modulate lower-level temporal dynamics through prediction error.

Design implication:

- Supports the idea that prediction should be layer-local and hierarchical.
- Current v1 keeps coupling simple in time rather than depth: layer `l` predicts the next-token target from the same layer's EMA encoder; do not introduce bidirectional recurrent inference yet.

#### NextLat

Source: <https://arxiv.org/abs/2511.05963>

Core idea:

- Adds next-latent prediction to next-token training.
- Trains latent states to be predictive of future latent states, adding a transition-like inductive bias without changing inference architecture.

Design implication:

- Relevant to our LM-head addition.
- Key similarity: the current predictor is an auxiliary next-latent loss and does not change inference computation directly.
- The older residual-delta-injection variant remains an ablation idea, not the current implementation.

### Tier 5: Generative and Bidirectional JEPA Variants

#### D-JEPA

Source: <https://arxiv.org/abs/2410.03755>

Core idea:

- Reinterprets JEPA as a masked image modeling / generalized next-token strategy and combines it with diffusion or flow-matching losses for generation.

Design implication:

- Useful conceptually because our model also has a token-level LM head.
- Do not mix diffusion/flow losses into text v1.

#### BiJEPA

Source: <https://arxiv.org/abs/2603.00049>

Core idea:

- Adds bidirectional JEPA prediction and cycle consistency.
- Notes symmetric prediction can produce representation explosion and uses norm regularization.

Design implication:

- Avoid bidirectional prediction in v1.
- If we later make layer `l` and `l+1` mutually predictive, add norm controls from the start.

## Main Belief Updates

### 1. EMA Is Useful But Not Sufficient

BYOL and I-JEPA justify EMA teachers. C-JEPA, VICReg, LeJEPA, and EB-JEPA all reinforce that collapse prevention must be measured explicitly.

Decision:

- Keep EMA because it is part of the architecture definition, but scope it to compressors only in the first pass.
- Add variance/effective-rank diagnostics immediately.
- Add variance regularization before scaling.

### 2. The LM Head Is Defensible

data2vec proves language latent prediction is a valid direction. LLM-JEPA directly supports combining language-model training with JEPA-style embedding-space objectives. NextLat supports auxiliary latent prediction alongside next-token training.

Decision:

- Keep the borrowed tokenizer path and LM head in v1.
- Test LM-only, JEPA-only, and combined objectives.

### 3. Residual Delta Injection Was The Novel Risk

Most JEPA/data2vec/BYOL systems use predictor heads for training, not as persistent inference-time residual updates. PredNet and predictive coding are the closest precedent for prediction-error dynamics, while NextLat is the closest language adjacent baseline.

Decision:

- Current implementation removed residual-delta injection from the base path. Treat it as a controlled ablation if revisited.

### 4. Teacher Target Placement Needs A Controlled Ablation

The older cross-depth target design said:

```text
y_l = CEbar_{l+1}(h_{l+1})
delta_target_l = y_l - stopgrad(z_l)
```

The current implementation instead says:

```text
y_l = CEbar_l(h_l_post_attn)
delta_target_l[:, t] = y_l[:, t+1] - z_l[:, t]
```

Literature does not directly settle whether the teacher should be a same-layer next-token compressor, a next-layer compressor, a projected next residual, a frozen teacher, or a projection head.

Decision:

- v1 target: EMA copy of the same layer compressor applied to `h_l_post_attn`, shifted by one token in the loss. Use `Identity` target projection by keeping one shared `K`.
- ablation A: target is normalized `h_{l+1}` projected to `K`.
- ablation B: target is frozen-teacher compression of `h_{l+1}`.
- ablation C: target is current student future compression `CE_{l+1}(h_{l+1})` with stop-gradient.

### 5. Borrow The Tokenizer

None of the literature argues that tokenizer sophistication is the first-order question for this architecture.

Decision:

- Borrow tokenizer/embedding machinery for v1.
- Keep model tests tensor-level where possible.

## Concrete Additions To The Implementation Plan

Add these from the literature pass:

1. Collapse metrics from the first smoke test:
   - per-layer target variance
   - per-layer prediction variance
   - effective rank
   - norm ratio `||pred|| / ||target||`

2. Delta checks:
   - auxiliary predictor delta norm stays finite
   - transition MLP updates the residual state independently from the auxiliary latent path

3. Objective ablations:
   - LM only
   - JEPA only
   - LM + JEPA
   - LM + JEPA with residual-delta injection as an ablation

4. Teacher ablations:
   - EMA same-layer next-token compressor
   - EMA next-layer compressor
   - projected next residual
   - frozen teacher later if EMA is unstable

5. Regularization progression:
   - diagnostics only
   - variance floor
   - VICReg-style variance/covariance
   - SIGReg

## Reading Priority For Implementation

Before coding:

1. I-JEPA
2. BYOL
3. data2vec
4. LLM-JEPA
5. VICReg
6. PredNet
7. NextLat

Before scaling:

1. LeJEPA
2. C-JEPA
3. V-JEPA
4. V-JEPA 2
5. Rethinking JEPA / SALT
6. EB-JEPA

Before trying bidirectional or generative extensions:

1. BiJEPA
2. D-JEPA
3. Dynamic Predictive Coding
