# Answer-Conditioned JEPA-SFT for Existing LMs

## Status

Planning document only. No implementation decisions in this file should be treated as final until validated by a small synthetic experiment.

## Goal

Define a supervised fine-tuning objective for an existing causal language model that augments standard next-token prediction with a JEPA-style local representation loss.

The core idea is:

- the **student** sees only the prompt/question
- an **EMA teacher** sees the prompt/question plus privileged supervised target text derived from the gold annotation
- at selected transformer layers, the student is trained to predict the teacher's answer-conditioned hidden states
- standard autoregressive loss remains the main behavioral anchor

This uses the gold answer as privileged training-time information to shape intermediate representations, without requiring pretraining infrastructure changes.

## Intended Model Target

Initial target model: `Qwen3.5-0.8B` as named in project discussion.

Assumption:

- this document treats the target as a small causal decoder-only transformer suitable for standard SFT
- exact hook names, block layout, and tokenizer details are intentionally left to implementation-time inspection

## Dataset Target

Planned dataset: `NVIDIA Nemotron-Post-Training-Dataset-v2`, assuming the local or downloaded JSONL schema is broadly consistent with the project discussion.

Expected fields:

- `problem`
- `qwen3-reasoning`
- `qwen3-solution`
- `reasoning` toggle
- `category`

Illustrative interpretation:

- `problem`: user-visible prompt
- `qwen3-reasoning`: teacher-generated chain-of-thought style trace
- `qwen3-solution`: final answer
- `reasoning`: whether the sample is intended to include an explicit reasoning trace
- `category`: task family metadata

Important:

- exact field names and formatting must be verified against the actual dataset before implementation
- this document treats the above schema as a planning assumption, not yet as a verified fact

## Motivation

Standard SFT supervises output tokens but does not directly supervise the model's internal state formation. This method asks:

> Can we improve SFT by training the model to form internal states that resemble the states it would have formed if it already knew the correct answer?

Potential benefit:

- better prompt-side internal organization before decoding
- stronger task-specific internal representations
- a cheap form of answer-conditioned self-distillation during SFT

Reason for skepticism:

- the next-token loss may already provide most of the useful signal
- direct hidden-state matching may add optimization burden without helping generation much
- this may produce only small gains or no gain at all

Conclusion:

- the idea is plausible enough for a small experiment
- it should be treated as an empirical question, not assumed to help

## Method Summary

For a supervised example:

- prompt/question tokens: `q`
- privileged target text tokens: `z*`
- emitted SFT target tokens: `y*`
- full teacher input: `x_t = [q; z*]`
- student JEPA input: `x_s = q`

Run two forward passes:

1. **Student path**
   - live model being fine-tuned
   - input: `q`
   - outputs hidden states at selected layers
   - participates in the local JEPA loss

2. **Teacher path**
   - EMA copy of the student
   - input: `q + z*`
   - no gradient
   - outputs answer-conditioned hidden states at selected layers

At each selected layer, train the student to predict the teacher representation for the prompt token positions only.

In addition, standard autoregressive SFT remains active on the emitted training target `y*`.

This document deliberately separates:

- the text seen by the student JEPA path
- the text seen by the teacher JEPA path
- the text used for the next-token supervised objective

These do not need to be identical.

## Nemotron-Specific Target Construction

For the Nemotron reasoning-style dataset, two privileged teacher conditioning modes are plausible.

### Teacher Mode A: `solution_only`

- `q = problem`
- `z* = qwen3-solution`

Interpretation:

- the teacher is conditioned only on the correct final answer
- the student is trained to form prompt-side hidden states that resemble the teacher's answer-aware states

This is the recommended first experiment.

### Teacher Mode B: `reasoning_plus_solution`

- `q = problem`
- `z* = <thought> qwen3-reasoning </thought> qwen3-solution`

Interpretation:

- the teacher is conditioned on the full visible reasoning trace plus final answer
- the student is trained toward thought-aware prompt representations

This is a second-phase ablation, not the default first experiment.

### Why `solution_only` Comes First

The reasoning trace is likely synthetic teacher-generated text, which introduces additional style and verbosity bias. Using only the final solution makes the privileged target simpler and more semantically grounded.

Working belief:

- `solution_only` is the cleaner test of whether privileged answer information helps
- `reasoning_plus_solution` tests whether explicit chain-of-thought conditioning adds anything beyond the final answer

## SFT Output Modes

The emitted supervised target `y*` should be treated as a separate design decision from the teacher conditioning text `z*`.

Two initial modes are worth planning for.

### Output Mode A: `solution_only`

Train the model to emit only the final answer:

```text
User: [problem]
Assistant: [solution]
```

### Output Mode B: `thought_then_solution`

Train the model to emit a reasoning trace followed by the final answer:

```text
User: [problem]
Assistant: <thought> [reasoning] </thought> [solution]
```

Important:

- the emitted target `y*` and the teacher conditioning target `z*` can differ
- for example, the model may be trained to emit `thought_then_solution` while the teacher JEPA path is conditioned only on `solution`

This separation should be preserved in the implementation spec.

## Key Design Choice

The student should not be forced to match teacher-only continuation token positions directly, because those positions do not exist in the student-only input stream.

Therefore the first version should match:

- teacher hidden states at the positions corresponding to the prompt tokens
- computed from the teacher forward pass on `q + z*`

Interpretation:

- the student learns what the prompt representation should look like if the correct answer were already integrated into the model's internal state

## Architecture Contract

### Student

- existing causal LM under SFT
- receives prompt tokens only
- local JEPA losses backpropagate through selected hidden states
- standard next-token loss remains active

### Teacher

- EMA copy of the student parameters
- receives prompt plus gold answer
- forward pass only
- stop-gradient everywhere
- updated after each training step with EMA

### Match Site

Preferred hook point:

- post-block residual stream for selected transformer blocks

Not preferred for v0:

- FFN output alone
- attention output alone
- layernorm internals

Reason:

- the block output is the cleanest stable hidden-state contract in a transformer

## Loss Definition

Let:

- `L` be the set of selected layers
- `h_s^l(q)` be the student hidden state at layer `l`
- `h_t^l(q | q+z*)` be the teacher hidden state at the prompt token positions, computed from the full input
- `P_l` be a small per-layer predictor head or projection

Per-layer local loss:

```text
L_local^l = Reg( P_l(h_s^l(q)), stopgrad(h_t^l(q | q+z*)) )
```

Total loss:

```text
L_total = L_ntp + lambda * sum_l w_l * L_local^l
```

Where:

- `L_ntp` is standard autoregressive next-token loss on the gold answer
- `L_ntp` is standard autoregressive next-token loss on the emitted target `y*`
- `lambda` controls the strength of the auxiliary objective
- `w_l` are optional per-layer weights
- `Reg` is a representation regression loss

## Recommended Regression Loss

Start with one of:

- cosine distance on normalized hidden states
- mean squared error on normalized hidden states

Preferred initial choice:

- cosine distance after layer-wise normalization

Reason:

- raw hidden-state magnitude can vary across layers and training phases
- cosine-style matching focuses more on direction than scale

## Predictor Head Choice

Two reasonable options:

1. Direct matching without a predictor
2. Small predictor `P_l` per selected layer

Recommended first choice:

- a lightweight predictor head per selected layer

Reason:

- direct equality constraints on raw hidden states are usually too rigid
- a predictor gives the student room to map prompt-only states into answer-conditioned target space

## Layer Selection

Do not attach losses to every layer in the first version.

Recommended v0:

- top one-third to one-half of transformer blocks

Reason:

- lower layers are likely to encode generic lexical and syntactic structure
- answer-conditioning pressure is more likely to be useful in middle and higher layers
- matching all layers risks over-regularizing the model

## Data Contract

Each training example should provide:

- prompt/question text
- gold solution text
- optional reasoning trace text
- reasoning-toggle metadata when available
- optional category metadata

Teacher input:

- serialized prompt plus privileged teacher text `z*`

Student input:

- prompt only

Important:

- no accidental answer leakage into the student path
- no matching on teacher-only continuation token positions in the first version
- mask local losses to prompt token positions only
- preserve a clean separation between `z*` and `y*`

## Why This Might Help

Possible mechanism:

- next-token loss teaches how to emit the answer
- local JEPA loss teaches how to internally organize the prompt state as if the answer were already understood

This could improve:

- internal task decomposition
- representation quality for short reasoning chains
- sample efficiency in small-model SFT

This is most plausible when:

- the task has a relatively canonical answer
- the dataset is high quality
- the model is small enough that extra representational guidance matters

## Why This Might Not Help

This method may fail for straightforward reasons:

1. Next-token loss may already be sufficient.
2. Hidden-state matching may be too indirect to improve output quality.
3. Gold answers may be only one of many valid continuations.
4. Synthetic reasoning traces may inject style bias rather than useful latent supervision.
5. Extra local losses may interfere with the base LM's useful representation geometry.
6. The teacher target may become too easy or too noisy, depending on EMA settings.

Working assumption:

- gains, if any, are likely modest
- a null result is plausible

## Risk Register

### Over-regularization

If `lambda` is too high, the model may optimize latent similarity instead of generation quality.

Mitigation:

- keep `lambda` small
- warm up the local loss after baseline SFT begins
- limit the number of matched layers

### Teacher Drift

If the EMA teacher changes too quickly, targets may be unstable.

Mitigation:

- use a high EMA decay
- consider updating teacher only once per optimizer step

### Open-Ended Targets

For tasks with many valid answers, the gold answer may impose an unnecessarily sharp internal target.

Mitigation:

- start on more deterministic instruction-following or short-answer data

### Synthetic CoT Bias

If the reasoning trace is generated by a stronger teacher model, JEPA may distill that trace style rather than generalizable competence.

Mitigation:

- start with `solution_only` teacher conditioning
- treat `reasoning_plus_solution` as an explicit ablation

### Representation Mismatch

Direct raw-state matching may be brittle.

Mitigation:

- normalize states
- use predictor heads
- start with cosine loss

## Minimal Experiment Spec

### Primary Question

Does privileged target-conditioned local representation matching improve SFT over a standard baseline on a small decoder-only model?

### Model

- `Qwen3.5-0.8B` as the first target model, pending implementation-time verification of exact model naming and APIs

### Baseline

- standard SFT with next-token loss only
- baseline should be measured for both `solution_only` and `thought_then_solution` output modes when practical

### Proposed Variant

- baseline plus EMA-teacher local JEPA loss on upper transformer blocks

### First Ablations

1. Baseline SFT with `solution_only` output
2. Baseline SFT with `thought_then_solution` output
3. JEPA-SFT with `solution_only` teacher conditioning
4. JEPA-SFT with `reasoning_plus_solution` teacher conditioning
5. JEPA-SFT with direct match instead of predictor
6. JEPA-SFT with different `lambda`
7. JEPA-SFT on fewer versus more layers

### Recommended First Matrix

If experiments must stay small, prioritize this order:

1. Baseline SFT on `problem -> solution`
2. JEPA-SFT with `teacher_mode = solution_only`
3. Baseline SFT on `problem -> <thought> reasoning </thought> solution`
4. JEPA-SFT with `teacher_mode = reasoning_plus_solution`

This isolates three separate questions:

- does visible reasoning-target SFT help at all
- does privileged answer-conditioned latent supervision help
- does reasoning-trace-conditioned latent supervision help beyond final-answer conditioning

### Metrics

- training stability
- validation next-token loss
- downstream instruction-following eval chosen at implementation time
- ablation on generation quality, not just hidden-state loss reduction

### Success Bar

Treat this as worth continuing only if one of the following holds:

- measurable validation improvement over baseline
- equal validation quality with better sample efficiency
- clearly better behavior on a task slice where answers are canonical

If none hold, drop the method.

## Expected Implementation Shape

This method should be implementable on top of an existing HF-style SFT stack by adding:

- an EMA teacher wrapper
- hidden-state interception hooks for selected layers
- prompt-position alignment and masking logic
- per-layer local predictors
- auxiliary loss aggregation

It should also preserve explicit dataset-to-training transforms:

- raw Nemotron record -> prompt text `q`
- raw Nemotron record -> privileged teacher text `z*`
- raw Nemotron record -> emitted target text `y*`

It should not require:

- new pretraining infrastructure
- masked prediction pipelines
- architecture changes to the core transformer blocks

## Recommended v0 Decisions

If a first prototype is built, the default settings should be:

- teacher: EMA copy of student
- student input: prompt only
- teacher input: prompt plus `qwen3-solution`
- local match positions: prompt tokens only
- local match layers: upper half of blocks
- hook point: post-block hidden states
- predictor: lightweight per-layer projection
- local regression: cosine on normalized states
- global anchor: standard next-token loss
- local weight: small
- initial dataset mode: `solution_only` teacher conditioning before any reasoning-trace conditioning

## Open Questions

1. Should the predictor be shared across layers or separate per layer?
2. Should the student still run on the full sequence for NTP while local loss only uses prompt positions, or should the training stack be fully split into prompt-only student and full-sequence teacher views?
3. Does the method help only on deterministic tasks and hurt open-ended conversational SFT?
4. Does matching a pooled latent work better than dense tokenwise prompt matching?
5. Is EMA actually better than a frozen teacher initialized from the base checkpoint?
6. Does `reasoning_plus_solution` help, or does it mostly distill synthetic chain-of-thought style?
7. Should the `reasoning` on/off flag change the teacher conditioning rule per sample?

## Decision

Proceed only as a scoped experiment.

This is not yet justified as a generally useful SFT paradigm. The idea is interesting because it uses supervised data to construct privileged latent targets, but the burden of proof is empirical and the expected gains may be small. For the Nemotron-style dataset, the cleanest first test is to use the final solution as the privileged teacher text and treat reasoning-trace conditioning as a later ablation.
