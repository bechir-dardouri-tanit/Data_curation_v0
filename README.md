# medrl

Post-training stack for a bilingual (EN + FR) medical reasoning LLM built on `Qwen/Qwen3.5-9B`.

Pipeline: **eval harness → baselines → data → ablations → SFT → preference → RL → on-policy
distillation**. The evaluation harness is built first and gates everything downstream.

## Design commitments

- **Every RL algorithm is a config point.** CISPO, SAPO, GSPO, GMPO, Dr.GRPO, GDPO, clip-cov /
  kl-cov and the rollout-mismatch corrections (TIS / Geo-RS) are selected by name in YAML, never
  by editing a loss function. Golden tests assert our config translation reproduces the reference
  loss exactly.
- **The eval scorers *are* the RL reward.** `medrl.rl.rewards` imports `medrl.eval.verifiers` and
  `medrl.eval.scorers`, so a rubric cannot mean one thing in training and another in evaluation.
- **Scale portability is structural.** `cluster=<preset>` is the only thing that differs between
  2×H100 and 8×H100+; no stage config hardcodes a world size.
- **Everything is content-addressed.** Each stage emits a manifest of input hashes, code SHA and
  config hash, making re-runs cache hits and ablation arms reproducible.

## Layout

| Path | Purpose |
|---|---|
| `src/medrl/core/` | config schema, hashing, manifests, registry |
| `src/medrl/model/` | arch invariants, checkpoint surgery (vision/MTP strip) |
| `src/medrl/eval/` | Inspect AI tasks, scorers, verifiers, vLLM serving |
| `src/medrl/data/` | schema, dedup, decontamination, teacher generation |
| `src/medrl/train/` | SFT and preference optimization |
| `src/medrl/rl/` | typed façade over verl: recipes, rewards, samplers, diagnostics |
| `src/medrl/ablation/` | declarative sweeps with decision rules |
| `src/medrl/analysis/` | bootstrap CIs, minimum detectable effect, promotion gate |

Code lives on `/`; **all weights, datasets and artifacts live on `/scratch`** (`MEDRL_SCRATCH`).

## Quickstart

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
just test          # CPU-only unit tests
just probe         # assert Qwen3.5-9B architecture invariants against the Hub
just convert       # emit the text-only checkpoint + logit-equivalence check   (GPU)
just eval          # run the decision-grade benchmark set                      (GPU)
```

## Target model

`Qwen/Qwen3.5-9B` is **dense** (no MoE), 32 layers in a 24 `linear_attention` (Gated DeltaNet) :
8 `full_attention` hybrid, `vocab_size` 248320 with untied embeddings, mRoPE, and a 1-layer MTP
head. See `src/medrl/model/probe.py` for the invariants we assert and why each one matters.
