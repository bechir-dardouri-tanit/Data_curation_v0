#!/usr/bin/env bash
# B7 pilot: proportional 8% sample through the CPU stages.
set -u
source /root/medrl/.venv/bin/activate
export HF_HOME=/scratch/medrl/.cache/huggingface HF_HUB_CACHE=/scratch/medrl/.cache/huggingface/hub
export HF_DATASETS_CACHE=/scratch/medrl/.cache/datasets
export HF_TOKEN="${HF_TOKEN:?export HF_TOKEN first}"
LOG=/scratch/medrl/curation/logs/pilot-cpu.log
RUN=pilot-b7
mkdir -p /scratch/medrl/curation/logs

python - << 'PYEOF' > "$LOG" 2>&1
import time
from medrl.curation import registry
from medrl.curation.stages import answers, concept, dedup_lex, decontam_ngram, normalize, structural
from medrl.curation.store import save_manifest, seal_manifest

RUN = "pilot-b7"
# Proportional 8% with a 2k floor (small high-value reasoning pools keep signal).
LIMITS = {
    "ii_medical_reasoning_sft": 175_000, "finemed_sft": 58_000,
    "chatdoctor_healthcaremagic": 9_000, "generalthought_biology": 34_000,
    "medical_r1_distill": 2_000, "m23k_tokenized": 2_000, "medreason": 2_600,
    "huatuo_o1_reasoning": 3_600, "finemed_dpo": 2_600, "ii_medical_rl": 2_000,
    "chatdoctor_rl": 2_000,
}
BENCHES = ["medqa","medmcqa","mmlu_pro_health","medxpertqa_text","mediqal",
           "healthbench_hard","medcalc","mmlu_pro","healthbench","ifeval"]

def run(stage_name, fn, *a, **kw):
    t0 = time.monotonic()
    m = seal_manifest(fn(*a, **kw))
    save_manifest(m)
    print(f"[{stage_name}] rows_in={m.rows_in} rows_out={m.rows_out} wall={m.wall_s}s", flush=True)
    if stage_name == "02_structural":
        print("  flag_rates:", {k: round(v["_all"], 4) for k, v in m.flag_rates.items()}, flush=True)
    return m

run("00_registry", registry.run_registry, RUN)
run("01_normalize", normalize.stage_entry, RUN, limit_per_source=LIMITS)
run("02_structural", structural.stage_entry, RUN)
run("03_dedup", dedup_lex.stage_entry, RUN)
run("04_decontam_ngram", decontam_ngram.stage_entry, RUN, benchmarks=BENCHES)
run("08_concept", concept.stage_entry, RUN,
    input_dir=f"/scratch/medrl/curation/{RUN}/04_decontam_ngram")
run("09_answers", answers.run_answers, RUN, input_stage="08_concept")
print("PILOT-CPU-CHAIN DONE", flush=True)
PYEOF
echo "EXIT:$?" >> "$LOG"
