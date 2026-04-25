# Autoresearch Experiment Log

Last updated: 2026-04-25

This is the human-readable log for what has been tried. The row-level source of truth is still `autoresearch/results.tsv`; this file records the conclusions and the context that a TSV row does not capture.

## Current Best

Best compact run so far:

- Run: `compact16m-5k-topthird-mean-t1-t4-20260425`
- Config: `sweep_configs/compact_16m_topthird_mean_t1_t4.yaml`
- Training: `5000` steps, `5.12M` tokens, CUDA bf16, compile on
- Active auxiliary layers: JEPA blocks `6, 7, 8` only
- Effective auxiliary dropout fraction: `0.098600`
- Result: `val_bpb=2.111987`
- Final eval: `loss_lm=3.627675`, `loss_jepa=1.883511`, `loss_sigreg=8.630859`
- Note: this replaced top-third+10% dropout (`2.114118`) as the current best compact BPB. Margin is small (`0.002131` BPB), but consistently strongest among tested 5k compact runs so far.

Second-best compact result to compare against:

- Run: `compact16m-5k-topthird-drop10-20260424`
- Config: `runs/autoresearch/compact16m-5k-topthird-drop10-20260424/autoresearch_config.yaml`
- Result: `val_bpb=2.114118`
- Final eval: `loss_lm=3.631544`, `loss_jepa=2.351347`, `loss_sigreg=7.630859`

Best prior compact reference:

- Run: `compact16m-5k-rowwise-auxlight`
- Config: `sweep_configs/compact_16m.yaml`
- Result: `val_bpb=2.376269`
- Settings: `lambda_jepa=0.2`, `beta_sigreg=0.005`, warmups `1000`, EMA `0.99`

LM-only compact control:

- Run: `compact16m-5k-rowwise-lmonly`
- Result: `val_bpb=2.434470`
- Conclusion: raw LM is worse than aux-light by `0.058201` BPB and worse than top-third JEPA by `0.317279` BPB at the same token budget.

Model-size note:

- The current compact config name is historical. Instantiated from `sweep_configs/compact_16m.yaml`, it has `22,888,320` total parameters.
- Trainable student parameters: `21,525,504`
- Frozen EMA target parameters: `1,362,816`

## Evaluation And Harness Fixes

What we tested:

- Autoresearch wrapper calls `scripts.train_intertwined_hjepa`; it is not a forked trainer.
- BPB evaluation was corrected to use loader-shifted labels directly. The dataloader already provides `input_ids=t[0..L-1]` and `labels=t[1..L]`; shifting again silently scored `t -> t+2` and inflated BPB.
- The wrapper records `val_bpb`, eval losses, train losses, aux dropout statistics, throughput, memory, run dir, checkpoint path, and config path in `results.tsv`.
- The wrapper can now override JEPA layer controls from the CLI:
  - `--jepa-dropout-rate`
  - `--auxiliary-layer-start`
  - `--auxiliary-layer-stride`

Current rule:

- Use `val_bpb` as the keep/discard metric.
- Use JEPA/SIGReg diagnostics to explain failures, not to override BPB.

## Architecture Ablations

Early 125M-style architecture tests are detailed in `autoresearch/architecture_debug_2026-04-17.md`. The main lesson was that the old cross-layer target and residual injection path created long-run auxiliary blow-up.

Tried:

- Baseline next-layer target path: auxiliary losses grew badly while LM stayed near `~6.0`.
- `Projector(delta)` instead of `Projector(z + delta)`: did not fix blow-up.
- `Projector(0.1 * (z + delta))`: small 5k improvement, not enough.
- Residual target instead of post-attention target: materially delayed blow-up.
- EMA teacher next-block attention target: worse.
- Same-block residual target: improved short-horizon JEPA, but still unstable longer.
- Skipping layer-0 aux in that older setup: worse.
- Stop-gradient through `z` inside the old projector path: strong 5k false positive, worse by 10k.
- Encoding only attention delta or input residual: worse than encoding full post-attention residual.
- Predicting target directly instead of residual delta: worse than delta prediction.

Promoted architecture cleanup:

- Current target is same-layer EMA CE, shifted one token in the loss:
  - `target_z_l = EMA_CE_l(h_l_post_attn)`
  - `delta_l[:, t] -> target_z_l[:, t+1] - z_l[:, t]`
- The JEPA predictor is now auxiliary-only.
- The residual stream uses a separate transition MLP, not `Projector(z + delta)`.

Discarded or low-priority directions:

- Strict layer-local auxiliary backprop regressed badly: `val_bpb=4.249308` at 5k on the 83M auxlighter baseline.
- Direct residual-delta injection should stay an ablation, not the base path.

## 83M Future-Token Runs

Useful references:

- `ft83m-auxlighter-wd0-5k-20260418-0911`: `val_bpb=3.485299`
- `ft83m-auxlighter-wd0-5k-compile-confirm-20260418-2051`: `val_bpb=3.452238`
- `full-20260418-215929`: clean post-fix beta `0.005`, `val_bpb=3.479250`
- `full-20260418-221321`: local aux-gradient variant, `val_bpb=4.249308`, discard
- `future-token-83m-auxmid-wd0-5k-20260419`: `val_bpb=4.254167`, discard

Conclusion:

- For the larger future-token setup, aux-light settings were better than heavier auxiliaries, but the compact 16M path became more productive for fast BPB iteration.

## Compact 16M Runs

Heavy auxiliary settings:

- `compact16m-5k-rowwise`: `val_bpb=3.487933`
- `compact16m-alt-5k-rowwise`: `val_bpb=3.490147`
- `compact16m-5k-rowwise-save`: `val_bpb=3.489307`
- Conclusion: `lambda_jepa=1.0`, `beta_sigreg=0.1`, EMA `0.996` was far too heavy and caused collapse-like behavior.

Aux-light settings:

- `compact16m-5k-rowwise-auxlight`: `val_bpb=2.376269`
- Key settings: `lambda_jepa=0.2`, `beta_sigreg=0.005`, warmups `1000`, EMA `0.99`
- Conclusion: major win over heavy aux and LM-only.

Lighter aux continuation and rerun:

- `compact16m-5k-rowwise-auxlighter2k`: `val_bpb=2.388355`
- Settings: `lambda_jepa=0.15`, longer warmups `2000`
- Conclusion: continuation looked plausible, but fresh 5k run was worse than aux-light. Keep `lambda_jepa=0.2`, warmups `1000` as the compact default until a real architecture change beats it.

LM-only:

- `compact16m-5k-rowwise-lmonly`: `val_bpb=2.434470`
- Conclusion: auxiliary learning helps this compact model at the current budget.

Top-third JEPA:

- `compact16m-5k-topthird-jepa-20260424`: `val_bpb=2.117191`
- Active layers: `6, 7, 8`; layers `0..5` had zero JEPA/SIGReg losses.
- Conclusion: concentrating JEPA/SIGReg on the top third helped more than supervising every JEPA block.

Top-third JEPA plus 10% auxiliary dropout:

- `compact16m-5k-topthird-drop10-20260424`: `val_bpb=2.114118`
- Effective dropout fraction: `0.098600`
- Final eval: `loss_lm=3.631544`, `loss_jepa=2.351347`, `loss_sigreg=7.630859`
- Conclusion: this is the historical top-third baseline, and set the reference for 10% dropout in this branch.

Top-third same-layer mean target over future span t+1..t+4:

- `compact16m-5k-topthird-mean-t1-t4-20260425`: `val_bpb=2.111987`
- Config: `sweep_configs/compact_16m_topthird_mean_t1_t4.yaml`
- Change tested: target for layers `6,7,8` replaced with same-layer average of future compressed states
  over `t+1 : t+5` (`t+1`, `t+2`, `t+3`, `t+4`).
- Effective dropout fraction: `0.098600`
- Final eval: `loss_lm=3.627675`, `loss_jepa=1.883511`, `loss_sigreg=8.630859`
- Throughput: `16700.368212` tokens/sec, wall time `317.295727` seconds
- Conclusion: this is the current best compact result. Despite slightly higher SIGReg than top-third baseline, the BPB gain is best-in-class.

Top-third same-layer mean target with normalized cosine:

- `compact16m-5k-topthird-mean-t1-t4-cosine-20260425`: `val_bpb=2.123667`
- Config: `sweep_configs/compact_16m_topthird_mean_t1_t4_cosine.yaml`
- Change tested: JEPA loss is normalized cosine on the same top-third+mean-span target (`t+1..t+4`).
- Effective dropout fraction: `0.098600`
- Final eval: `loss_lm=3.647982`, `loss_jepa=0.787334`, `loss_sigreg=5.210938`
- Conclusion: very low JEPA/SIGReg magnitudes but worse BPB than delta mean target; does not beat best compact yet.

Top-third projected residual-stream delta target over mean future span t+1..t+4:

- `compact16m-5k-topthird-residual-delta-t1-t4-20260425`: `val_bpb=2.123149`
- Config: `sweep_configs/compact_16m_topthird_residual_delta_t1_t4.yaml`
- Change tested: project `h_final[t+1:t+5]` mean before forming delta target in residual space.
- Effective dropout fraction: `0.098600`
- Final eval: `loss_lm=3.646943`, `loss_jepa=0.897406`, `loss_sigreg=4.559570`
- Conclusion: not competitive with best current mean target; weaker LM auxiliary tradeoff.

Top-third same-layer mean target with stronger auxiliary weights:

- `compact16m-5k-topthird-mean-t1-t4-auxup-20260425`: `val_bpb=2.120223`
- Config: `sweep_configs/compact_16m_topthird_mean_t1_t4_auxup.yaml`
- Change tested: same current-best top-third mean target (`layers 6,7,8`, horizon `t+1..t+4`, 10% aux dropout), but increased auxiliary weights from `lambda_jepa=0.2`, `beta_sigreg=0.005` to `lambda_jepa=0.25`, `beta_sigreg=0.0075`.
- Effective dropout fraction: `0.098600`
- Final eval: `loss_lm=3.641958`, `loss_jepa=1.818791`, `loss_sigreg=7.882812`
- Throughput: `21686.934701` tokens/sec, wall time `295.211317` seconds
- Runtime note: FA-4 attempted and failed internally, then the attention wrapper fell back to PyTorch SDPA; the run completed normally.
- Conclusion: heavier auxiliary weighting regressed versus the current best by `0.008236` BPB. Keep the current-best `lambda_jepa=0.2`, `beta_sigreg=0.005` unless another architecture changes the loss balance.

Top-third + 25% dropout vs alternatives:

- `compact16m-5k-topthird-drop25-20260425`: `val_bpb=2.117229`, effective dropout fraction `0.247800`
  - Final eval: `loss_lm=3.636989`, `loss_jepa=2.367961`, `loss_sigreg=8.066406`
  - Throughput: `22368.689305` tokens/sec, wall time `288.335386` seconds
- `compact16m-5k-layers6-8-mean-t1-t4-drop25-20260425`: `val_bpb=2.118200`, effective dropout fraction `0.247800` (only layers `6,8`)
  - Config: `sweep_configs/compact_16m_layers_6_8_mean_t1_t4_drop25.yaml`
  - Final eval: `loss_lm=3.638445`, `loss_jepa=1.585545`, `loss_sigreg=5.869141`
  - Throughput: `22745.394200` tokens/sec, wall time `316.397741` seconds
- `compact16m-5k-layers4-8-mean-t1-t4-drop20-20260425`: `val_bpb=2.116224`, effective dropout fraction `0.197000` (layers `4,5,6,7,8`)
  - Config: `sweep_configs/compact_16m_layers_4_8_mean_t1_t4_drop20.yaml`
  - Final eval: `loss_lm=3.635137`, `loss_jepa=3.050288`, `loss_sigreg=16.685547`
  - Throughput: `21657.626851` tokens/sec, wall time `317.225610` seconds
- Conclusion: wider/deeper active layer sets and higher dropout did not beat the 10% top-third mean-target path; they are currently lower-priority baselines.

Multi-horizon grouped JEPA:

- `compact16m-5k-horizon-groups-t2-20260425`: `val_bpb=2.126356`
- Config: `sweep_configs/compact_16m_horizon_groups_t2.yaml`
- Groups: layers `3,4` same-layer `t+2..t+5`; layers `5,6` same-layer `t+2..t+9`; layers `7,8` final-layer `t+2..t+12`
- Effective dropout fraction: `0.098600`
- Final eval: `loss_lm=3.652208`, `loss_jepa=4.933363`, `loss_sigreg=19.546875`
- Throughput: `20507.106141` tokens/sec, wall time `329.740579` seconds
- Conclusion: this was worse than top-third + 10% dropout by `0.012238` BPB. The six-layer grouped horizon setup is too much auxiliary pressure at this budget, even though it ran correctly and stayed stable.

Direct final-layer span state prediction:

- `compact16m-5k-topthird-finalspan-state-20260425`: `val_bpb=2.130347`
- Config: `sweep_configs/compact_16m_topthird_final_span_state.yaml`
- Change tested: top-third layers `6,7,8` directly aligned their compressed state to a final-layer EMA future summary over `t+2..t+9` instead of predicting a delta to the target.
- Effective dropout fraction: `0.098600`
- Final eval: `loss_lm=3.659471`, `loss_jepa=1.937029`, `loss_sigreg=44.171875`
- Throughput: `22414.918147` tokens/sec, wall time `292.154382` seconds
- Conclusion: this was worse than top-third + 10% delta prediction by `0.016229` BPB. The lower JEPA loss did not translate to better LM BPB, and SIGReg pressure rose sharply. The implementation was committed for history and then reverted from the active code.

Final-layer future teacher without SIGReg:

- `compact16m-5k-topthird-final-future-nosigreg-t1-t4-20260425`: `val_bpb=2.143707`
- Config: `sweep_configs/compact_16m_topthird_final_future_nosigreg_t1_t4.yaml`
- Change tested: layers `6,7,8` predicted a final-layer future teacher over `t+1..t+4`; SIGReg was disabled for this teacher objective (`beta_sigreg=0.0`).
- Effective dropout fraction: `0.098600`
- Final eval: `loss_lm=3.682296`, `loss_jepa=0.738716`, `loss_sigreg=0.000000`
- Throughput: `24074.658894` tokens/sec, wall time `326.998018` seconds
- Conclusion: clean final-layer future teacher without SIGReg was substantially worse than the current same-layer mean target by `0.031720` BPB. Low JEPA loss here was not a useful signal; this objective made LM worse and should stay reverted from the active path.

Split content+dynamics latents:

- `compact16m-5k-topthird-mean-t1-t4-split-latents-20260425`: `val_bpb=2.206675`
- Historical config: `sweep_configs/compact_16m_topthird_mean_t1_t4_split_latents.yaml` during the run; removed from active sweep configs after rollback.
- Change tested: current-best top-third mean setup (`layers 6,7,8`, horizon `t+1..t+4`, 10% aux dropout), with the latent budget split into `content_dim=112` and `dynamics_dim=112` while preserving the old total `compressed_dim=224`.
- Contract:
  - `z_content` feeds SIGReg / stable representation pressure.
  - `z_dynamics` feeds JEPA prediction, EMA targets, and predicted deltas.
  - Non-split configs retain the old single-latent behavior.
- Final eval: `loss_lm=3.790255`, `loss_jepa=1.240986`, `loss_sigreg=4.382812`
- Throughput: `20069.002832` tokens/sec, wall time `345.869230` seconds
- Verification: focused model tests passed (`46 passed`) and `autoresearch/train.py --validate-only --profile smoke` accepted the split config.
- Diagnostic note: SIGReg was much lower than the current best, but LM loss regressed badly. Dynamics latent scale became unstable in active layers; late-run `z_variance` reached hundreds to thousands and `delta_norm` reached thousands on layers `6,7,8`.
- Conclusion: same-total-width 50/50 split latents are a clear discard for this setup, worse than current best by `0.094688` BPB. The result suggests the dynamics branch became too underconstrained or too narrow after removing SIGReg pressure from the JEPA latent. The implementation was rolled back from active code; the run result remains in `results.tsv` for history.

Zero-gated JEPA residual adapter:

- `compact16m-5k-topthird-mean-t1-t4-gated-adapter-20260425`: `val_bpb=2.124057`
- Config: `sweep_configs/compact_16m_topthird_mean_t1_t4_gated_adapter.yaml`
- Change tested: current-best top-third mean setup (`layers 6,7,8`, horizon `t+1..t+4`, 10% aux dropout), plus a zero-initialized gated residual adapter from normalized predicted future latent `norm(z + delta)` back into the residual transition on layers `6,7,8`.
- Contract:
  - Gate is parameterized as `tanh(raw_gate)` and initialized to exactly `0.0`.
  - At gate `0.0`, the adapter preserves the old residual update exactly.
  - LM gradients can move the gate while it is closed; once the gate opens, LM gradients also flow into the adapter, compressor, and predictor unless `jepa_residual_adapter_stop_gradient=true`.
- Final eval: `loss_lm=3.648525`, `loss_jepa=1.936831`, `loss_sigreg=9.730469`
- Effective dropout fraction: `0.098600`
- Throughput: `22694.915117` tokens/sec, wall time `298.439852` seconds
- Diagnostic note: gates stayed very close to zero throughout the run, e.g. late train logs were roughly within `[-0.0042, 0.0037]` on active layers. Adapter update norms were large, but the learned scalar gates kept the residual contribution tiny.
- Verification:
  - Focused pytest suite passed: `46 passed`.
  - `autoresearch/train.py --validate-only --profile smoke` accepted the config.
  - Tiny smoke training completed.
  - Deterministic numeric probes passed for future-span means, exact adapter residual math, zero-gate preservation, real compact-config shapes/layer routing, and LM gradient flow through the opened adapter.
- Conclusion: implemented and verified, but this config is a discard for BPB. It regressed versus current best by `0.012070` BPB and did not meaningfully exploit the residual adapter at the 5k budget.

Still unrun but now supported:

- alternating blocks such as one-based 4/6/8 via `auxiliary_layer_start=3`, `auxiliary_layer_stride=2`
- weighted or lower-SIGReg horizon ablations, since direct final-layer span state prediction caused high SIGReg loss at unchanged `beta_sigreg`

## Optimization And Runtime Tests

CUDA / compile:

- Local GPU: RTX PRO 4500 Blackwell.
- PyTorch needed to be upgraded to `2.9.1` CUDA `13.0` wheels for `sm_120`.
- CUDA training now compiles by default.
- `--no-compile` was removed as an option; keep compile enabled for real CUDA runs.

Flash attention:

- `flash-attn-4==4.0.0b9` can import after dependency fixes.
- Direct CUTE runtime smoke failed with an internal `NoneType` trait error on this GPU.
- The repo wrapper therefore attempts FA-4 and falls back to PyTorch SDPA.
- Forced PyTorch SDPA backends were locally smoke-tested and worked: FlashAttention SDPA, EfficientAttention SDPA, cuDNN Attention, and Math.

TorchAO float8:

- Rowwise float8 converted the linear layers successfully.
- Synthetic compact benchmark:
  - baseline bf16 compile: about `20,507 tok/s`
  - TorchAO rowwise float8: about `18,790 tok/s`
- Conclusion: rowwise float8 was about `8.4%` slower for the compact model. Do not use it as the default compact training path.

## Current Repro Commands

Top-third JEPA plus 10% aux dropout:

```bash
uv run -- python autoresearch/train.py \
  --profile full \
  --config sweep_configs/compact_16m.yaml \
  --parameter-golf-root /path/to/parameter-golf \
  --jepa-dropout-rate 0.1 \
  --auxiliary-layer-start 6 \
  --auxiliary-layer-stride 1 \
  --run-name compact16m-topthird-drop10
```

Alternating one-based 4/6/8 JEPA blocks:

```bash
uv run -- python autoresearch/train.py \
  --profile full \
  --config sweep_configs/compact_16m.yaml \
  --parameter-golf-root /path/to/parameter-golf \
  --jepa-dropout-rate 0.1 \
  --auxiliary-layer-start 3 \
  --auxiliary-layer-stride 2 \
  --run-name compact16m-jepa-4-6-8-drop10
```

## Current Beliefs

- Best BPB currently comes from compact 16M aux-light with JEPA/SIGReg only on top JEPA blocks `6, 7, 8`, 10% dropout, and mean-span target (`t+1..t+4`).
- Supervising all JEPA blocks is not automatically better.
- Auxiliaries are useful, but too much auxiliary weight is harmful.
- Layer selection is the strongest recent gain; 10% auxiliary dropout is a consistent gain on top-third setups.
- 25% auxiliary dropout is too high or at least not beneficial at the current 5k-step budget.
- Increasing the current-best auxiliary weights to `lambda_jepa=0.25`, `beta_sigreg=0.0075` regressed BPB.
- Broad six-layer multi-horizon supervision regressed; if trying horizons again, keep the active layer count closer to the top-third winner.
- Directly matching top-third states to final-layer future spans regressed at unchanged SIGReg weighting.
- Final-layer future teacher without SIGReg also regressed; low JEPA loss from that objective did not mean better LM.
- Split content+dynamics latents with a 50/50 same-total-width split regressed badly; the content branch got low SIGReg but the dynamics branch became scale-unstable and LM BPB worsened.
- Float8 and FA-4 are not current wins in this environment; PyTorch bf16 compile with SDPA fallback is the practical path.
