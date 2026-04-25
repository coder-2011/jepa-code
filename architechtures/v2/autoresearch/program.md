# autoresearch program

This repo is not the original standalone autoresearch setup anymore. The stable entrypoint is `autoresearch/train.py`, but it wraps the real Intertwined H-JEPA trainer in the parent repo.

## Read First

Before making any experiment change, read:

1. `AGENTS.md`
2. `README.md`
3. `learning.md`
4. `intertwined_hjepa.py`
5. `scripts/train_intertwined_hjepa.py`
6. `scripts/inspect_checkpoint.py`
7. `autoresearch/train.py`
8. `autoresearch/README.md`

## Scope

Primary target:

- optimize `val_bpb`

Supporting diagnostics:

- `eval_loss`
- `eval_loss_lm`
- `eval_loss_jepa`
- `eval_loss_sigreg`
- memory
- throughput
- JEPA dropout statistics
- profiler evidence when performance, memory, compile behavior, or GPU utilization is part of the question

## Hard Rules

- Never edit `jepa/`.
- Never edit `lejepa/`.
- Do not fork the real trainer into `autoresearch/train.py`.
- Keep `autoresearch/train.py` as a wrapper/orchestrator only.
- Preserve the current Intertwined H-JEPA tensor and loss contracts.
- Prefer small, reviewable changes.

## Editable Files

Preferred allowlist:

- `autoresearch/train.py`
- `intertwined_hjepa.py`
- `intertwined_hjepa.yaml`
- `scripts/train_intertwined_hjepa.py`
- focused tests under `test/`

You may update `autoresearch/README.md` and `autoresearch/program.md` when the workflow changes, but those are not where model quality comes from.

## Setup

1. Validate the experiment harness first:

```bash
uv run -- python autoresearch/train.py \
  --validate-only \
  --parameter-golf-root /path/to/parameter-golf
```

2. Initialize the baseline with a smoke run after any structural wrapper change:

```bash
uv run -- python autoresearch/train.py \
  --profile smoke \
  --parameter-golf-root /path/to/parameter-golf \
  --description "baseline smoke" \
  --status baseline
```

3. For real experiments, run the wrapper and redirect logs:

```bash
uv run -- python autoresearch/train.py \
  --profile full \
  --parameter-golf-root /path/to/parameter-golf \
  --description "<experiment description>" \
  --status pending \
  > autoresearch/run.log 2>&1
```

## Experiment Loop

1. Inspect the current git state.
2. Make one focused change.
3. Run a smoke check if the change is structural.
4. Use profilers when they can answer the question better than logs alone.
5. Run the full wrapper entrypoint.
6. Read the final summary block or `autoresearch/run.log`.
7. Use `val_bpb` as the keep/discard metric.
8. Record the run in `results.tsv` through the wrapper.

## Profiling

- Use NVIDIA tools when they can clarify bottlenecks or correctness-adjacent performance issues.
- Prefer Nsight Systems for end-to-end timelines: CPU/GPU overlap, dataloader stalls, kernel launch gaps, compile overhead, synchronization, and step-time structure.
- Prefer Nsight Compute for kernel-level questions: matmul/attention efficiency, memory bandwidth, occupancy, tensor core use, and whether a suspected CUDA kernel is the real bottleneck.
- Use PyTorch profiling when it is enough: operator attribution, memory snapshots, autograd hotspots, and quick local iteration before heavier NVIDIA profiler runs.
- Keep profiler runs small and explicit. Use short smoke profiles, focused step ranges, and clear output paths; do not profile long autoresearch runs by default.
- Include profiler evidence in the experiment notes or result discussion whenever it affects a design decision.

## Decision Rule

- Lower `val_bpb` is better.
- If BPB is meaningfully better, keep the change.
- If BPB is flat or worse, prefer the simpler implementation.
- Use JEPA and SIGReg diagnostics to understand regressions, not to override BPB.

## Failure Handling

- If the run crashes, inspect the traceback in `autoresearch/run.log`.
- Fix simple integration bugs quickly.
- If the idea is fundamentally broken, record it as `crash` or `discard` and move on.

## Notes

- `results.tsv` is intentionally untracked by git.
- The wrapper requires a saved final checkpoint because BPB is computed post-run from the actual trained model.
- Do not convert this back into a duplicate one-file trainer.
