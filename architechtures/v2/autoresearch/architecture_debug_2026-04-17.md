# Architecture Debug 2026-04-17

## Goal

Debug the long-run auxiliary-loss blow-up in the 125.5M architecture on real cached FineWeb data.

## Common Run Setup

- Config: `sweep_configs/best_wd0_125m.yaml`
- Device: `cuda`
- Dtype: `bfloat16`
- Compile: `on`
- Batch size: `8`
- Sequence length: `128`
- Dataset: real cached FineWeb `sp1024`
- Train corpus for ablations: 4 real train shards exposed at `/tmp/fineweb10B_sp1024_4shards`
- Validation: 1 real val shard

## Baseline Problem

Observed on the unmodified architecture:

- `loss_lm` stays near `~6.0`
- `loss_jepa` and `loss_sigreg` grow steadily
- layer 0 blows up first
- `delta_norm_layer_0`, `z_std_mean_layer_0`, and `grad_norm` rise much faster than later layers

Interpretation before ablations:

- this looks architectural, not just an optimizer or logging issue
- the likely failure mode is a feedback loop in the JEPA target path

## Variant Log

### 1. Baseline

Status: completed

Result summary:

- Reproduced the blow-up pattern.
- Eval trajectory:
  - 5k: `eval/loss ~= 19.06`, `eval/loss_jepa ~= 3.47`, `eval/loss_sigreg ~= 95.99`
  - 10k: `eval/loss ~= 30.18`, `eval/loss_jepa ~= 8.74`, `eval/loss_sigreg ~= 154.41`
  - 15k: `eval/loss ~= 88.44`, `eval/loss_jepa ~= 40.36`, `eval/loss_sigreg ~= 420.79`
  - 20k: `eval/loss ~= 119.14`, `eval/loss_jepa ~= 46.79`, `eval/loss_sigreg ~= 663.27`

### 2. `delta_only`

Change:

- Replace `Projector(z + delta)` with `Projector(delta)`.

Status: completed

Measured 5k eval:

- `eval/loss ~= 20.35`
- `eval/loss_lm ~= 6.018`
- `eval/loss_jepa ~= 4.011`
- `eval/loss_sigreg ~= 103.21`

Conclusion:

- Did not fix the problem.
- Slightly worse than baseline at 5k.

### 3. `scaled_z_plus_delta`

Change:

- Replace `Projector(z + delta)` with `Projector(0.1 * (z + delta))`.

Status: completed

Measured 5k eval:

- `eval/loss ~= 18.67`
- `eval/loss_lm ~= 6.015`
- `eval/loss_jepa ~= 3.108`
- `eval/loss_sigreg ~= 95.43`

Conclusion:

- Small improvement at 5k.
- Not enough to explain the long-run instability.

### 4. `residual_target`

Change:

- Build the JEPA target from the next residual state instead of the next post-attention state.
- Concretely: use `EMA_CE_{l+1}(x_{l+1})` rather than `EMA_CE_{l+1}(h_{l+1,post_attn})`.

Status: completed

Measured eval trajectory:

- 5k:
  - `eval/loss ~= 17.58`
  - `eval/loss_lm ~= 5.993`
  - `eval/loss_jepa ~= 2.227`
  - `eval/loss_sigreg ~= 93.58`
- 10k:
  - `eval/loss ~= 25.44`
  - `eval/loss_jepa ~= 6.262`
  - `eval/loss_sigreg ~= 131.79`
- 15k:
  - `eval/loss ~= 52.36`
  - `eval/loss_jepa ~= 22.77`
  - `eval/loss_sigreg ~= 235.75`
- 20k:
  - `eval/loss ~= 87.21`
  - `eval/loss_jepa ~= 33.12`
  - `eval/loss_sigreg ~= 480.57`

Conclusion:

- Clear improvement over baseline.
- The blow-up is still present, but it is materially delayed and reduced.
- This strongly suggests the target-construction path is the main architectural issue.

## Current Belief

The highest-signal finding so far is:

- changing the residual injection formula alone does not solve the issue
- changing the JEPA target source helps much more

That points to the partial-teacher target path as the main architectural problem:

```text
target_z_l = EMA_CE_{l+1}(student_post_attn_{l+1})
```

This is only partially teacher-generated because the target modules are EMA, but the hidden state feeding them is still student-produced and moves with the student residual dynamics.

## Promoted Change

The first architectural fix promoted into the real codebase is:

- switch JEPA target construction from next post-attention state to next residual state

Code-level contract after the change:

```text
target_z_l = EMA_CE_{l+1}(x_{l+1})
```

For the top JEPA block:

```text
target_z_top = output_target_compressor(output_target_norm(h_final))
```

Reason for promotion:

- it was the strongest improvement among the tested variants
- it directly addresses the partial-teacher target problem
- it is a smaller and cleaner architectural change than trying to re-EMA the whole next block immediately

Direct-code verification after promotion:

- 5k direct run on the modified code:
  - `eval/loss ~= 19.44`
  - `eval/loss_lm ~= 5.994`
  - `eval/loss_jepa ~= 2.709`
  - `eval/loss_sigreg ~= 107.39`

Interpretation:

- the direct code change still improves JEPA loss relative to the original baseline at 5k
- it is not as strong as the earlier monkeypatch residual-target run
- this means the target-path issue is real, but more architectural cleanup is still likely needed

### 5. `teacher_next_block_attention_target`

Change:

- Keep the residual target source.
- Add EMA copies of the next block's attention norm and attention module.
- Build the target as `EMA_CE_{l+1}(x_{l+1} + EMA_Attn_{l+1}(EMA_AttnNorm_{l+1}(x_{l+1})))`.

Status: completed

Measured 5k eval:

- `eval/loss ~= 20.60`
- `eval/loss_lm ~= 5.994`
- `eval/loss_jepa ~= 3.654`
- `eval/loss_sigreg ~= 109.44`
- `val_bpb ~= 4.2119`

Conclusion:

- This is worse than both the baseline residual-target code and the earlier monkeypatch residual-target result.
- The main regression is layer-0 JEPA, so adding more teacher-owned next-block structure did not solve the feedback loop in the way expected.
- This direction should not stay in the main code.

### 6. `same_block_residual_target`

Change:

- Keep the residual target source.
- Stop changing representation basis at the same time.
- Build the target as `EMA_CE_l(x_{l+1})` instead of `EMA_CE_{l+1}(x_{l+1})`.

Status: completed

Measured 4-shard 5k eval:

- `eval/loss ~= 17.14`
- `eval/loss_lm ~= 5.997`
- `eval/loss_jepa ~= 0.980`
- `eval/loss_sigreg ~= 101.56`
- `val_bpb ~= 4.2146`

Conclusion:

- Best 5k total loss so far on the direct code path.
- JEPA improved dramatically relative to both the baseline code and the earlier residual-target code that still switched basis to layer `l + 1`.
- SIGReg is still not ideal, but it is better than the failed teacher-next-block-attention variant and better than the earlier direct residual-target code.
- This suggests the basis shift itself is part of the architectural instability: predicting the next residual in the next layer's representation basis is likely too strong and too self-referential.

Measured 4-shard 10k eval:

- `eval/loss ~= 26.56`
- `eval/loss_lm ~= 6.002`
- `eval/loss_jepa ~= 3.544`
- `eval/loss_sigreg ~= 169.95`
- `val_bpb ~= 4.2185`

Updated conclusion:

- Still better than the original 4-shard baseline at 10k (`~26.56` vs `~30.18` total loss).
- This is a real delay/reduction of the blow-up, not just a 5k cosmetic win.
- It is not a full fix: layer 0 still dominates the JEPA term by 10k and SIGReg keeps climbing.

## Current Best Direction

The strongest direct-code result so far is:

```text
target_z_l = EMA_CE_l(x_{l+1})
```

Interpretation:

- using the next residual state still helps
- keeping the target in the current layer's CE basis helps even more
- the inter-layer basis change appears to be a second architectural stressor on top of the original target-source problem

Next check:

- run this variant longer to see whether it merely improves the first 5k or actually delays the long-run auxiliary blow-up

Note:

- An initial 5k run accidentally used the profile default `train_shards=1`, so its result is not comparable to the earlier 4-shard ablations.
- That run looked promising (`eval/loss ~= 17.79`, `eval/loss_jepa ~= 0.638`), but it should be ignored for architecture ranking until rerun on 4 shards.

## Fast 1k Sweep

To iterate faster, the next set of architectural probes used:

- `max_steps: 1000`
- `log_every: 1000`
- `eval_every: 1000`
- the same 4-shard real-data setup as above

### 7. `same_block_residual_target` anchor

Measured 1k eval:

- `eval/loss ~= 28.58`
- `eval/loss_lm ~= 5.996`
- `eval/loss_jepa ~= 0.560`
- `eval/loss_sigreg ~= 220.06`
- `val_bpb ~= 4.2112`

Conclusion:

- This is the 1k anchor for the fast sweep.

### 8. `same_block_post_attn_target`

Change:

- Keep the current-layer CE basis.
- Switch the target source back to the next post-attention state.
- Concretely: `EMA_CE_l(h_{l+1,post_attn})`.

Measured 1k eval:

- `eval/loss ~= 23.20`
- `eval/loss_lm ~= 6.000`
- `eval/loss_jepa ~= 0.669`
- `eval/loss_sigreg ~= 165.18`
- `val_bpb ~= 4.2139`

Conclusion:

- Worse than the residual-source anchor.
- This isolates target source as a real part of the fix: the same-block basis alone is not enough.

### 9. `same_block_residual_target_scaled_injection`

Change:

- Keep `EMA_CE_l(x_{l+1})`.
- Scale the injected residual update to `Projector(0.1 * (z_l + delta_l))`.

Measured 1k eval:

- `eval/loss ~= 23.92`
- `eval/loss_lm ~= 5.998`
- `eval/loss_jepa ~= 0.484`
- `eval/loss_sigreg ~= 174.25`
- `val_bpb ~= 4.2124`

Conclusion:

- JEPA improved a bit, but SIGReg worsened enough that total loss lost to the anchor.
- Smaller injection does not pair well with the current best target path.

### 10. `uniform_same_block_residual_target`

Change:

- Keep `EMA_CE_l(x_{l+1})` for non-top layers.
- Remove the special output-target encoder for the top JEPA block.
- Use the same top-layer rule as the other JEPA blocks: `EMA_CE_top(h_final)`.

Measured 1k eval:

- `eval/loss ~= 23.52`
- `eval/loss_lm ~= 5.995`
- `eval/loss_jepa ~= 0.477`
- `eval/loss_sigreg ~= 170.27`
- `val_bpb ~= 4.2101`

Conclusion:

- Also worse than the anchor.
- The special output-target encoder should stay for now.

### 11. `same_block_residual_target_skip_layer0_aux`

Change:

- Keep the current best target path: `EMA_CE_l(x_{l+1})`.
- Skip JEPA and SIGReg losses for the first JEPA block only.
- Keep the rest of the architecture unchanged.

Measured 1k eval:

- `eval/loss ~= 32.17`
- `eval/loss_lm ~= 6.004`
- `eval/loss_jepa ~= 0.479`
- `eval/loss_sigreg ~= 256.62`
- `val_bpb ~= 4.2169`

Conclusion:

- Worse than the 1k anchor.
- Removing layer-0 auxiliary pressure lowers JEPA trivially but lets SIGReg worsen enough that total loss gets worse.
- The remaining instability is not fixed by simply muting the first block.

### 12. `same_block_residual_target_delta_only`

Change:

- Keep the current best target path: `EMA_CE_l(x_{l+1})`.
- Change the injected residual update from `Projector(z_l + delta_l)` to `Projector(delta_l)`.

Measured 1k eval:

- `eval/loss ~= 26.08`
- `eval/loss_lm ~= 5.996`
- `eval/loss_jepa ~= 0.687`
- `eval/loss_sigreg ~= 193.77`
- `val_bpb ~= 4.2112`

Conclusion:

- Better than the 1k anchor on total loss.
- JEPA got a bit worse, but SIGReg improved enough that the overall objective improved.
- This is the first post-anchor variant in the fast sweep that clearly beats the current best direction, so it deserves a longer 5k check.

Measured 5k eval:

- `eval/loss ~= 22.27`
- `eval/loss_lm ~= 5.998`
- `eval/loss_jepa ~= 2.102`
- `eval/loss_sigreg ~= 141.59`
- `val_bpb ~= 4.2146`

Updated conclusion:

- This wins at 1k but loses badly by 5k.
- The delta-only injection looks like a short-horizon false positive rather than a real architectural fix.
- It should not replace the current best same-block residual target.

### 13. `same_block_residual_target_stopgrad_z_update`

Change:

- Keep the current best target path: `EMA_CE_l(x_{l+1})`.
- Keep the value-space update `Projector(z_l + delta_l)`.
- Stop gradient through the `z_l` branch inside the projector input: `Projector(z_l.detach() + delta_l)`.

Measured 1k eval:

- `eval/loss ~= 22.10`
- `eval/loss_lm ~= 5.994`
- `eval/loss_jepa ~= 0.388`
- `eval/loss_sigreg ~= 157.07`
- `val_bpb ~= 4.2100`

Conclusion:

- Best 1k result in the sweep so far.
- This beats both the anchor and the delta-only variant while keeping LM flat.
- The result suggests the projector's live `z` gradient path is part of the remaining short-horizon instability, even if the value contribution of `z` itself is still useful.

Measured 5k eval:

- `eval/loss ~= 16.41`
- `eval/loss_lm ~= 5.995`
- `eval/loss_jepa ~= 0.972`
- `eval/loss_sigreg ~= 94.37`
- `val_bpb ~= 4.2123`

Updated conclusion:

- New best 5k result in the sweep.
- This beats the previous best same-block residual-target code on both total loss and SIGReg while keeping JEPA in the same rough range.
- The strongest current belief is that the remaining instability is not just about which target the model chases, but also about the projector turning the live `z` branch into an overly strong gradient route.

Measured 10k eval:

- `eval/loss ~= 37.36`
- `eval/loss_lm ~= 6.003`
- `eval/loss_jepa ~= 8.656`
- `eval/loss_sigreg ~= 226.79`
- `val_bpb ~= 4.2192`

Updated conclusion:

- This is another short-horizon false positive.
- It wins strongly at 5k but degrades badly by 10k, ending up worse than the plain same-block residual target.
- The projector's live `z` gradient path matters, but fully stopping it is too aggressive for longer training.

### 14. `encode_attention_delta_only`

Change:

- Keep the current best target/update path.
- Change the compressed state to summarize only `attn_out` instead of the full post-attention residual.
- Concretely: `z_l = CE(attn_out)` instead of `z_l = CE(x_{l,post_attn})`.

Measured 1k eval:

- `eval/loss ~= 29.06`
- `eval/loss_lm ~= 5.998`
- `eval/loss_jepa ~= 1.135`
- `eval/loss_sigreg ~= 219.02`
- `val_bpb ~= 4.2129`

Conclusion:

- Worse than the current best anchor.
- Encoding only the attention novelty throws away too much useful residual context for this architecture.

### 15. `encode_input_residual_state`

Change:

- Keep the current best target/update path.
- Change the compressed state to summarize the incoming residual state `x_l` instead of the post-attention residual `x_{l,post_attn}`.

Measured 1k eval:

- `eval/loss ~= 16.70`
- `eval/loss_lm ~= 6.029`
- `eval/loss_jepa ~= 0.575`
- `eval/loss_sigreg ~= 100.86`
- `val_bpb ~= 4.2333`

Conclusion:

- Auxiliary losses look much better, but the LM side gets worse.
- This likely means the compressed state is losing too much token-prediction-relevant information when attention is moved fully outside the encoded representation.

### 16. `encode_half_attention_update`

Change:

- Keep the current best target/update path.
- Change the compressed state to summarize `x_l + 0.5 * attn_out` instead of the full post-attention residual.

Measured 1k eval:

- `eval/loss ~= 26.05`
- `eval/loss_lm ~= 5.998`
- `eval/loss_jepa ~= 0.671`
- `eval/loss_sigreg ~= 193.67`
- `val_bpb ~= 4.2123`

Conclusion:

- Better than the pure input-state encoding, but still worse than the current best post-attention encoding.
- The sweep so far points to full post-attention residual as the strongest `z` source among these simple representation choices.

### 17. `predict_target_directly`

Change:

- Keep the current best target/update path.
- Change JEPA so the predictor tries to match `target_z` directly instead of the residual delta `target_z - z`.

Measured 1k eval:

- `eval/loss ~= 26.51`
- `eval/loss_lm ~= 5.995`
- `eval/loss_jepa ~= 0.768`
- `eval/loss_sigreg ~= 197.25`
- `val_bpb ~= 4.2102`

Conclusion:

- Worse than the current best residual-delta contract.
- The residual prediction target still looks like the better predictor contract in this architecture.

Measured 5k eval:

- `eval/loss ~= 16.41`
- `eval/loss_lm ~= 5.995`
- `eval/loss_jepa ~= 0.972`
- `eval/loss_sigreg ~= 94.37`
- `val_bpb ~= 4.2123`

Updated conclusion:

- New best 5k result so far on the real 4-shard setup.
- This beats the prior best same-block residual target on total loss, JEPA loss, and SIGReg.
- The clearest current hypothesis is that the projector should still see `z` as a value offset, but should not backpropagate through the raw `z` branch.

Measured 10k eval:

- `eval/loss ~= 37.36`
- `eval/loss_lm ~= 6.003`
- `eval/loss_jepa ~= 8.656`
- `eval/loss_sigreg ~= 226.79`
- `val_bpb ~= 4.2192`

Updated conclusion:

- Another short-horizon false positive.
- This is clearly better than the current best at 5k, but clearly worse by 10k.
- Detaching `z` in the projector delays the instability briefly, then makes the later blow-up worse.

### 11. `same_block_residual_target_delta_only`

Change:

- Keep `EMA_CE_l(x_{l+1})`.
- Change the injected update to `Projector(delta_l)` instead of `Projector(z_l + delta_l)`.

Measured 1k eval:

- `eval/loss ~= 27.52`
- `eval/loss_lm ~= 5.999`
- `eval/loss_jepa ~= 0.758`
- `eval/loss_sigreg ~= 207.43`
- `val_bpb ~= 4.2135`

Conclusion:

- Worse than the anchor.
- The better target path does not rescue `delta_only`; the original `z_l + delta_l` injection should stay for now.
