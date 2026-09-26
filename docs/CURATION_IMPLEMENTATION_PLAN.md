# Curation Pipeline — Implementation Plan

**Status:** approved blueprint, awaiting build kickoff · **Version:** 1.0 · **Date:** 2026-09-26
**Scope:** S0–S15 data curation for the medrl medical-LLM program (~3.27M starting rows, EN+FR),
feeding the 5-phase training curriculum (Pool B/R/P → SFT → DPO → polish).
**Ground truth inputs:** repo gap analysis (32/32 verified findings), measured vLLM throughput on
this 2×H100 node, live PyPI/HF research (2026-09-26). Every external claim below carries its check date.

---

## 0. Design principles (non-negotiable)

1. **Flags, not deletes.** Stages never drop rows; they set typed flags. Deletion is a query.
   Every filtering decision stays auditable and reversible; mixtures become SQL.
2. **One canonical row type.** `CorpusItem` with *all* downstream columns reserved from day one
   (S1 contract). No mid-pipeline migrations.
3. **Pure stage contracts.** Every stage is `(Parquet dataset, StageConfig) → (dataset, StageManifest)`.
   The manifest is the audit trail; reports generate *from* manifests, never hand-written.
4. **Extend the existing stack.** `src/medrl/data/{dedup,decontam,filters,sources}.py` and
   `src/medrl/eval/scorers/judge.py` are wrapped and parameterized — no parallel implementations.
   The eval-time contamination preflight stays as the final gate after S15.
5. **Durability as an invariant.** Heavy artifacts → `/scratch/medrl/curation/`; light records
   (manifests, registry, reports, mixture SQL) → `experiments/curation/` **committed + pushed at every
   stage boundary**. `/scratch` has been wiped twice; any stage resumes from manifest + Parquet snapshot.
6. **Cheapest-first, pilot-gated.** Rule-based stages before GPU stages; every GPU stage measured on the
   8% pilot before full-scale commitment; generation concurrency pinned at the measured knee (C=192).

---

## 1. Decision log (all checked 2026-09-26)

### 1.1 Models

| Role | Decision | Why | Confidence |
|---|---|---|---|
| **Embedding (S5)** | **`Qwen/Qwen3-Embedding-4B`** — 2560-d Matryoshka (32–2560), last-token pooling, Apache-2.0. Fallback: `Qwen/Qwen3-Embedding-0.6B` (1024-d) | MMTEB-v2 Mean(Task) **69.45 vs bge-m3 59.56** (+9.9); cross-lingual EN↔FR is inside that margin; native vLLM support since 0.8.5; ~450M prefill tok ≈ 1.5–2.5 h on 2×H100 (0.6B: 20–40 min) | high |
| Embedding (held for pilot A/B) | `microsoft/harrier-oss-v1-0.6b` (MIT, 69.0 MMTEB-v2) | Near-parity at 1/7 cost but **no vLLM 0.27 recipe yet** (SGLang only), no MRL — head-to-head on the pilot, not default | med |
| Embedding rejected | bge-m3 (plan original), nemotron-embed-8b (non-commercial licence), jina-v5 (CC-BY-NC), Qwen3-VL-Embedding (multimodal) | dominated or licence-incompatible | high |
| **Judge (S11)** | `Qwen/Qwen3.5-4B`, thinking **off** at server level | 10.2k out tok/s measured; binary-criteria judge infra exists; fast-tier caveat → calibrate against a bigger judge on ~2k rows before trusting as filter | high |
| **Difficulty (S12)** | `Qwen/Qwen3.5-9B`, thinking **on**, budgets pinned from `decision_grade.yaml` (n=8, max_tokens 10240, think 8192) | repo pass@8 precedent; measured 7.3k tok/s; truncation-retry rule mandatory (measured 12–20% think-truncation at these budgets on medcalc-class rows) | high |
| **Generation (S14)** | `Qwen/Qwen3.8-27B`, temp 0.7 top-p 0.95, N=4–6, MTP spec-decode after A/B | measured 2.5k tok/s; budget by kept traces; 1k-prompt yield probe first | high |
| **LID (S1)** | `fasttext lid.176.bin` (checksum-pinned) + `GlotLID V3` second opinion for prob<0.90 or <20 chars | ~112k sents/s/core → whole corpus in minutes; langdetect is 480× slower (rejected); fastText has documented EN-bias → never hard-drop, audit 1,000/lang | high |

### 1.2 Libraries (exact pins; live PyPI checks 2026-09-26)

```toml
# pyproject [project.optional-dependencies] additions
curation = [
  "duckdb==1.5.5",            # 2026-07-22; storage/query engine; parquet built-in; vss ext = experimental, NOT used for ANN
  "datatrove[io]~=0.10.0",    # 2026-08-13; CPU-stage executors (LocalPipelineExecutor) for S2–S4 filters
  "datasketch==2.0.0",        # 2026-07-05; active again; 2.0 fixes permutation over-estimation; MinHash-LSH
  "usearch==2.26.2",          # 2026-08-31; ANN index, f16 quantization + mmap; faiss-cpu==1.15.1 documented fallback
  "ladybug==0.20.4",          # ⚠️ KUZU IS DEAD (repo archived 2025-10-10, last 0.11.3) — ladybug is the maintained
                              #     community continuation ("formerly known as Kuzu", same embedded Cypher)
  "xxhash>=3.5,<4.0",
  "fasttext-community>=0.9.2",# lid.176.bin runner (model file pinned by checksum, not by pip)
]
entity = [
  "medcat==2.9.0",            # ⚠️ MedCAT v1 archived 2025-07-28 — v2 lives as cogstack-nlp; primary EN linker
  "edsnlp==0.23.0",           # 2026-09-22; most active lib in the stack; FR linking WITHOUT UMLS (see §1.3)
  "scispacy==0.6.2",          # optional ensemble only; requires_python <3.13 → caps python
]
```

```toml
# pyproject [project] — required
requires-python = ">=3.12,<3.13"   # <3.13 cap exists solely for scispacy 0.6.2
```

**Rejected:** `distilabel` (last release 2025-01-28, 20 months stale; authors moved on) — rejection
sampling stays on the hardened in-repo asyncio loop. `duckdb-vss` (experimental) — ANN is usearch's job.

### 1.3 Knowledge/terminology backbone (S7/S8/S10/S13)

| Layer | Decision |
|---|---|
| **EN backbone** | **UMLS** (file the free UTS licence application **today**; approval hours–2 weeks; release 2025AB, ~3.45M concepts). Linker: **MedCAT v2** primary, scispaCy optional ensemble |
| **FR, day one (no licence needed)** | **EDS-NLP ready-mades**: `eds.drugs` (BDPM brand+INN→ATC), `eds.cim10` (ICD-10 FR), `eds.adicap`, plus `eds.terminology` (simstring) for loaded dictionaries and trainable `eds.span_linker` for fuzzy linking. BDPM flat files (free, ANSM) as drug source; **skip WHODrug** (commercial) |
| **FR, licence track** | **SNOMED CT FR via ANS National Release Center** (France member since 2022-11-30; MLDS affiliation) — join `SCTID → UMLS CUI`. Public catalog SMT (smt.esante.gouv.fr) for CIM-10/CCAM/LOINC. **"Merlin"/"PPM" from the original plan are unverifiable — removed** |
| **Fallback if UMLS stalls** | **OMOP (Athena) + DIY French layer**: ⚠️ verified correction — `CONCEPT_SYNONYM` is **English-only** in current OMOP; we inject French ourselves by joining SNOMED FR RF2 descriptions on `SCTID = concept_code`, CIM-10 FR via `concept_relationship 'Maps to'`, BDPM names onto RxNorm/ATC. KG layer stays behind a thin interface so ladybug ⇄ DuckDB-graph-schema (nodes/edges + recursive CTE, optional `duckpgq`) swap is one config flag |
| **Version discipline** | Every Parquet build stamps: UMLS release, OMOP bundle download date, SNOMED FR RF2 version, CIM-10 FR version, BDPM snapshot date. Reproducibility of linking = pinned terminologies |

### 1.4 Serving (vLLM 0.27.1 pin; upstream is 0.30.0 — upgrade *after* current eval sweep)

```bash
# S5 embedding (one server for the whole corpus pass)
vllm serve Qwen/Qwen3-Embedding-4B --runner pooling --max-model-len 512 --port 8101
# legacy --task embed still accepted; DP across both H100s; client batches 64–256 texts

# S11 judge — thinking OFF at server level (no per-request kwarg needed)
vllm serve Qwen/Qwen3.5-4B --reasoning-parser qwen3 --language-model-only \
  --default-chat-template-kwargs '{"enable_thinking": false}' --max-model-len 16384 --port 8102

# S12 difficulty — thinking ON, big context, optional MTP spec-decode (A/B before the 219 h run)
vllm serve Qwen/Qwen3.5-9B --reasoning-parser qwen3 --language-model-only \
  --max-model-len 32768 --port 8103
#   optional: --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
#   per-request cap: sampling param thinking_token_budget; client reads message.reasoning
#   APC is default-on: submit all k repeats of one item ADJACENTLY to share prefill

# S14 generation
vllm serve Qwen/Qwen3.8-27B --reasoning-parser qwen3 --language-model-only \
  --max-model-len 16384 --port 8104   # + MTP after A/B (low-concurrency, long outputs: best spec-decode case)
```

* `Answer: X` extraction constraint: `extra_body {"structured_outputs": {"regex": "Answer:[ ]*([A-E])"}}`
  (guided_\* fields were removed in vLLM 0.12 — the renamed API is structured_outputs/xgrammar).
* **Security:** every port localhost-bound/firewalled — CVE-2026-90878 (/v1/chat/completions) affects 0.27.1.
* All GPU stages run through ONE `curation/serving.py` gateway: AIMD adaptive concurrency (not fixed 192),
  `/metrics` scraping per run, resume-aware seeded loop (the eval `generate.py` pattern), generation
  manifest (model, vLLM version, seed base) beside every JSONL.

### 1.5 Measured compute budget (this 2×H100 node; C=192 knee; decode-bound arithmetic)

| Stage | Full-scale (100%) | 8% pilot |
|---|---|---|
| S5 embed 3.27M × ~150 tok (4B model) | 1.5–2.5 h | ~10 min |
| S11 judge 600k × 3 binary axes | 7.4–9 h | 0.6 h |
| S12 pass@8 600k × 8 × ~1.2k tok @9B | **219 h ≈ 9.1 days** | 17.5 h |
| S14 yield probe 1k × N=6 @27B | 1.3 h | — |
| CPU stages (S1–S4, S8, S13, S15) | hours (single node, datatrove/duckdb) | minutes |

⚠️ **S12 alone exceeds the original plan's 210 node-hour total.** Decide at the pilot gate: pass@4
(~110 h) vs pass@8 on this node (9.1 days) vs downweighting the 1.0-band before labelling vs bigger node.

---

## 2. Repo layout

```
src/medrl/curation/
  __init__.py
  schema.py            # CorpusItem, Flags, StageManifest, SourceRecord (pydantic, frozen)
  registry.py          # S0 — acquisition, sha256, licence resolution, registry.jsonl
  runner.py            # stage-graph executor: resume, manifests, reports, experiments/ mirroring
  serving.py           # THE GPU gateway: server lifecycle, AIMD concurrency, retries, metrics, gen-manifests
  thresholds.py        # single source of truth for every numeric threshold (see §3.2)
  stages/
    normalize.py       # S1
    structural.py      # S2
    dedup_lex.py       # S3
    decontam_ngram.py  # S4
    embed.py           # S5
    decontam_sem.py    # S6
    linking.py         # S7 (+ handcheck.py harness)
    concept.py         # S8
    answers.py         # S9
    grounding.py       # S10
    judge.py           # S11
    difficulty.py      # S12 (+ adaptive top-up)
    coverage.py        # S13
    generate.py        # S14
    mixture.py         # S15
configs/curation/
  sources.yaml         # the 11 pools + 12 benchmark indexes: hf_id, revision pin, licence, expected rows
  pipeline.yaml        # stage graph, enabled stages, thresholds overrides
  judge_axes.yaml      # the 3 axis criterion-sets (binary, weighted)
  mixtures/            # phase1.sql ... phase5.sql (the mixture specs, executable)
experiments/curation/<run_id>/     # manifests, reports, registry copy — COMMITTED
/scratch/medrl/curation/<run_id>/  # parquet datasets, ANN index, KG db — heavy, never sole-copy
tests/curation/                    # unit per stage + golden-set integration + property tests
```

---

## 3. Core contracts

### 3.1 `CorpusItem` (the S1 schema — every column reserved now)

```python
class Flags(BaseModel, frozen=True):          # all defaults False/None; stages only ever SET
    f_empty: bool = False
    f_length: bool = False                    # token-length bounds violated
    f_truncated: bool = False                 # <think> without </think>
    f_repetition: bool = False                # any 50-char span repeated >4x
    f_lang: bool = False                      # lang not in {en, fr} per keep-rule
    f_encoding: bool = False                  # mojibake/replacement >0.1%
    f_refusal: bool = False
    f_dup_exact: bool = False
    f_dup_minhash: bool = False
    f_dup_semantic: bool = False
    f_dup_concept: bool = False
    f_contam_ngram: bool = False
    f_contam_semantic: bool = False
    f_contam_concept: bool = False
    f_answer_wrong: bool = False
    f_answer_right_reasoning_contradicts: bool = False
    f_kg_contradicted: bool = False

class CorpusItem(BaseModel, frozen=True):
    id: str                                   # "<source_id>:<row_id>"
    source: str
    lang: str = "en"                          # S1 LID writes; S2 keep-rule decides
    lang_score: float | None = None           # fastText prob (GlotLID re-score notes in meta)
    licence: str = "unknown"                  # resolved at S0; "unknown" + upstream-issue ref allowed
    redistributable: bool | None = None
    messages: list[Message]
    thinking: str | None = None
    tools: list | None = None
    answer: str | None = None                 # gold answer IF the source has one
    answer_type: Literal["mcqa","numeric","free_text","none"] = "none"
    meta: dict                                # source row, FineMed quality/complexity verbatim, snapshot stamps
    # reserved — filled by later stages, present from S1:
    flags: Flags = Flags()
    dup_of: str | None = None
    contam_benchmark: str | None = None
    embedding: list[float] | None = None      # fp16-quantized on write (bytes in parquet)
    cuis: list[str] = []                      # question CUIs (or SCTID/OMOP id per backbone)
    answer_cui: str | None = None
    kg_triples: list[tuple] = []
    support_frac: float | None = None
    unknown_frac: float | None = None
    q_coherence: int | None = None            # 0/1 binary axis outcomes (not 1-5)
    q_clinical: int | None = None
    q_format: int | None = None
    difficulty: float | None = None           # pass rate k/8 (post-top-up)
    difficulty_band: Literal["hold","rl","sft1","downsample"] | None = None
    n_covered_by: int = 0
```

Storage: one Parquet dataset per stage snapshot (`/scratch/medrl/curation/<run>/NN_<stage>/`),
`id` as sort key, written via pyarrow; queried with DuckDB views. Every `experiments/curation/<run>/NN_<stage>.manifest.json`
records: stage, config dump, input/output row counts, **flag-rate table per source**, input/output content
sha256, code_sha, wall time, thresholds effective.

### 3.2 `thresholds.py` — one module, every constant, one place to calibrate

| Constant | Initial | Calibration path |
|---|---|---|
| MINHASH perms / shingles / Jaccard | 128 / 5-gram word / **0.80** (plan) | exact-Jaccard verify pass over LSH candidates (repo does this) |
| NGRAM dedup / contam | 13-gram **word** (repo eval-side is word-level w/ char fallback; data-side is char — **reconcile to word-level, matching eval**) | — |
| SEM_DUP / SEM_CONTAM | 0.95 / 0.90 **from bge-m3 — DO NOT TRUST for Qwen3-Embedding** | recalibrate on B7 pilot: precision/recall on 1k labeled pairs + FR→EN recall@1 sanity (expect ≈1.0) |
| CONCEPT_JACCARD | 0.85 | S8 pilot hand-check |
| KG drop rule | drop only `contradicted`; `unknown` → coverage signal | — |
| S12 bands | 1.0→downsample · 0.5–0.9→sft1 · 0.1–0.4→rl · 0.0→hold | **route on intervals, not points**: adaptive top-up (+8–16 samples) for k∈{3,4,7,8} (n=8 Clopper-Pearson CIs span band boundaries); per-language thresholds (9B scores FR far below EN) |
| S12 budgets | max_tokens 10240, think 8192 (from `decision_grade.yaml`) | record think_completion_rate per shard; **retry incomplete-thinks, don't score as failures** |

---

## 4. Stage contracts (S0–S15)

**S0 `registry.py`** — For each of the 11 pools + 12 benchmark indexes: fetch with pinned HF revision,
sha256 of every downloaded file, row counts **before/after every filter** (the ChatDoctor "filtered to
112,165" was a no-op — record it as such), licence resolution (6 sources are untagged: record
`licence=unknown, upstream_issue=<url>, queried=<date>`), lang, has_cot (verified: 8/11 have traces;
ChatDoctor-HealthCareMagic and ChatDoctor-RL do NOT; II-Medical-Reasoning-SFT embeds reasoning inside
messages — S1 must extract), has_tools. Output: `registry.jsonl` → experiments/ (committed).
Huatuo-o1: **explicitly pick en (19.7k) + en_mix (24.9k) = 44.6k** and pin the choice.

**S1 `normalize.py`** — Map every source to `CorpusItem`; fastText LID on ≥200-char question
concatenation (never the shortest field), GlotLID second opinion; carry FineMed `quality`/`complexity`
verbatim into meta (do not re-derive); explicit token bounds (min 8, max 32768 — cluster max_model_len).

**S2 `structural.py`** — datatrove executors over the Parquet; the repo's length/language/ratio filters
PLUS four new: think-truncation (reuse eval `generate.py` _THINK_PREFILL convention), 50-char×>4
degenerate repetition, mojibake ratio, refusal templates. LID never hard-drops: `f_lang` + keep-rule query.

**S3 `dedup_lex.py`** — wrap `data/dedup.py`: exact sha256 (checksums finally computed — fixes the
declared-never-populated bug), 13-gram word-level, MinHash-LSH per §3.2. **New canonical keep-rule**
(lane-ranked): permissive licence → longer thinking trace → lower `source_id`; persist `dup_of`.
Medianash 2.0 permutation scheme.

**S4 `decontam_ngram.py`** — wrap `data/decontam.py` BenchmarkIndex; **question-text-only** indexing
(fixes the join-all-messages behavior); 12 benchmarks: 8 covered + 4 new loaders
(PubMedQA — ⚠️ test split lives on GitHub, not HF; medbullets test 124; DrBenchmark → enumerate member
datasets e.g. DiaMED test 154, cc-by-4.0; medagents-benchmark test+test_hard). FrenchMedMCQA needs the
HF token in this path. Keep `generate_negative_controls` canaries — they feed contamination_report.md.

**S5 `embed.py`** — Qwen3-Embedding-4B via the §1.4 server; embed **question only**; symmetric uses
(dedup/clustering) with NO instruction; decontam queries carry the EN retrieval instruction (card: 1–5%
at stake). Store fp16 2560-d in Parquet (~15 GB/3M); usearch HNSW on MRL-truncated 1024-d (~6.1 GB f16,
mmap); keep-rule re-scores candidates at full dim. Log the truncated-vs-full recall curve once on B7.

**S6 `decontam_sem.py`** — ANN self-join (dup) + benchmark-vectors join (contam); thresholds
recalibrated on B7 before any full run; flip the repo's `check_embeddings` default ON here (it stays
off in the legacy paths).

**S7 `linking.py`** — EN: MedCAT v2 + UMLS 2025AB (licence-gated). FR: EDS-NLP eds.drugs/cim10/adicap +
span_linker. Backbone-agnostic ids + a `backbone` column (cui | sctid | omop_id) so the S8 join keys
survive a UMLS→OMOP fallback swap. `handcheck.py`: stratified 300/lang sampling UI (JSONL out),
gates S7 sign-off; FR precision expected worse — measure it.

**S8 `concept.py`** — question-CUI-set + answer-CUI equality (dup) and Jaccard≥0.85 + answer-CUI match
vs eval items (contam). This is the only catch for FR translations of EN questions — the stage with no
substitute. DuckDB set-ops over the Parquet cuis columns.

**S9 `answers.py`** — split by `answer_type`: mcqa → structured_outputs regex + letter verify (eval
verifiers reused); numeric → verify_number rtol; **free_text/none → no f_answer_wrong** (they carry
FineMed quality labels or get S11's criteria instead — verified: ~95% of Pool B has no gold answer, the
~30%-removal estimate applies to the MCQA slice + Pool R only). The
`f_answer_right_reasoning_contradicts` flag: judge-authored criterion pair (states X / concludes ¬X).

**S10 `grounding.py`** — triple extraction (small-N prompts against the judge model, constrained JSON),
ladybug KG (interface: `KGBackend` with ladybug + DuckDB-graph implementations), supported/contradicted/
unknown; drop only contradicted; per-source `support_frac` → grounding_report.md.

**S11 `judge.py`** — batch job over `eval/scorers/judge.py` with the three NEW axis criterion-sets
(`configs/curation/judge_axes.yaml`, binary weighted); server-level thinking-off; calibration gate:
4B-vs-stronger-judge agreement ≥ target on ~2k rows before corpus-wide run; results persist to
`q_*` columns. 1.8M axis-calls ≈ 8–9 h measured-budget.

**S12 `difficulty.py`** — serve per §1.4; decision_grade budgets; k repeats submitted adjacently (APC);
think-incomplete → retry once at +4k budget, then mark `difficulty=None` + meta note (never counted as
failure); adaptive top-up for band-boundary k; bands per language; write `difficulty` + `difficulty_band`.
**219 h full-scale — pilot-gate decision required (pass@4 vs node upgrade vs 1.0-band downweight).**

**S13 `coverage.py`** — aggregate cuis/kg_nodes → coverage by node type, degree-weighted coverage,
high-degree-zero-coverage gap list → coverage_map.json + generation_targets.csv.

**S14 `generate.py`** — gap-targeted only; stock 27B; N=4–6; keep verifier-correct only (S9 machinery);
budget by kept traces; 1k-prompt yield probe before any full run (measured 1.3 h); FR gate =
**CUI-equivalence with the EN source** (not back-translation similarity). Generated rows re-enter at S2.
**No privileges.**

**S15 `mixture.py`** — phase SQL files (executable recipes, shipped with the model); assembly re-runs
S4/S6/S8 on the assembled set (recombination reintroduces contamination); final eval-time preflight gate.

---

## 5. Reports (all generated from manifests)

`registry.jsonl` (S0) · `dedup_report.md` (S3/S6/S8: rows per method + params) ·
`contamination_report.md` (S4/S6/S8 per-benchmark catch rates **including canary results**; publishes
with the model) · `grounding_report.md` (S10 source reliability ranking) · `coverage_map.json` +
`generation_targets.csv` (S13) · `mixture_spec.sql` (S15).

---

## 6. Build sequence & acceptance gates

| # | Deliverable | Effort | Acceptance gate |
|---|---|---|---|
| B0 | Commit+push repo backlog; `uv sync` new extras; UMLS **application filed**; SNOMED-FR ANS inquiry; download models (embed-4B, 9B, 27B) + 11 pools to scratch; HF token for FrenchMedMCQA path | ½ d | `medrl curate --help` runs; registry dry-run lists 11 sources w/ checksums |
| B1 | schema.py + registry.py + normalize.py (S0/S1) | 2 d | golden-set test: 1k rows, 11 sources → schema-valid Parquet + registry w/ real sha256; LID audit sample passes |
| B2 | structural.py (S2, 4 new filters) | 1 d | property tests: synthetic truncated/repeated/mojibake/refusal rows all flagged |
| B3 | dedup_lex.py (S3, keep-rule + params) | 1 d | planted-duplicate test: recall=1.0 on 500 planted pairs; dup_of integrity |
| B4 | decontam_ngram.py + 4 loaders (S4) | 2 d | canaries caught at expected rate; per-benchmark attribution correct; FrenchMedMCQA indexes w/ token |
| B5 | serving.py + embed.py + decontam_sem.py (S5/S6) | 2–3 d | FR→EN recall@1 sanity ≈1.0; thresholds recalibrated + written to thresholds.py |
| B6 | answers.py (S9 split-by-type) | 1–2 d | MCQA slice: ≥eval-grade extraction rates; free_text slice: no f_answer_wrong writes |
| B7 | **PILOT: 8% (~260k rows) through S1–S9, CPU mostly** | 3 d | **Go/No-Go**: real drop-rate table per stage per source; recalibrated thresholds; S12 scale decision made |
| B8 | judge.py + judge_axes.yaml (S11) | 2–3 d + 9 h GPU | calibration ≥ target on 2k labeled rows |
| B9 | difficulty.py (S12) | 1 d + pilot 17.5 h GPU | band populations sane; truncation-retry working; top-up triggered |
| B10 | linking.py + concept.py + grounding.py + coverage.py (S7/S8/S10/S13) | 1–2 wk (licence-gated) | 300/lang hand-check signed off; FR precision quantified |
| B11 | generate.py + mixture.py (S14/S15) | staged | yield probe measured; re-decontam catches recombination (canary test) |

---

## 7. Testing & quality

- **Unit:** every stage, synthetic inputs, flag-level assertions (property tests for filters: generate
  corrupted rows programmatically, assert exactly the right flag).
- **Golden-set integration:** fixed 1k-row seed corpus through the whole graph, checked into tests/
  fixtures; stage manifests must match recorded values (catches threshold drift instantly).
- **Canary discipline:** negative controls from `data/decontam.py` run in every S4/S6/S8 pass and in S15.
- **Resume tests:** kill-and-restart at every stage boundary in CI; manifest+snapshot resume must be exact.
- **Lint/typing:** ruff + mypy strict on `src/medrl/curation/`; pre-commit; 100% of thresholds imported
  from `thresholds.py` (grep-enforced via a lint test).

## 8. Open decisions (owner: you)

1. **UMLS application** — file today? (calendar critical path for S7/S8/S10/S13; EDS-NLP FR chain works
   day one regardless).
2. **S12 scale** — pass@4 (~110 h) vs pass@8 (9.1 d) vs node upgrade; decided at B7 with real band data.
3. **Unlicensed sources** — proceed with `licence=unknown` + upstream queries (recommended) or hold back
   the 6 untagged datasets (cuts 71% of Pool B).
4. **vLLM 0.30 bump** — after the current eval sweep, before S12's 219 h run (Mamba APC + MRV2 gains).
