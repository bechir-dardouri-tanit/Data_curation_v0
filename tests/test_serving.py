"""Deployment planning and config loading -- the CPU-testable eval skeleton."""

from __future__ import annotations

import pytest

from medrl.core.config import EvalConfig, ServingPattern
from medrl.eval.config_loader import load_eval_config
from medrl.eval.serving.vllm import ServePhase, plan_deployment
from medrl.eval.tasks.benchmarks import TASKS
from medrl.eval.tasks.prompts import MCQA_GRAMMAR, build_mcqa_user


@pytest.fixture()
def eval_config() -> EvalConfig:
    return load_eval_config("decision_grade")


def test_preset_resolves_cluster_and_benchmarks(eval_config: EvalConfig) -> None:
    assert eval_config.cluster.name == "local_2xh100"
    assert len(eval_config.benchmarks) == 9
    assert eval_config.cluster.num_gpus == 2


def test_sequential_plan_has_disjoint_phases(eval_config: EvalConfig) -> None:
    plan = plan_deployment(eval_config, policy_port=8100, judge_port=8101)
    assert plan.pattern is ServingPattern.SEQUENTIAL
    roles = [p.role for p in plan.phases]
    assert roles == ["policy", "judge"]  # judge strictly after policy
    assert plan.policy.port != plan.phases[1].port


def test_policy_command_carries_text_only_contract(eval_config: EvalConfig) -> None:
    argv = plan_deployment(eval_config, policy_port=8100).policy.command()
    joined = " ".join(argv)
    assert argv[0] == "vllm" and argv[1] == "serve"
    assert "--language-model-only" in argv
    assert "--reasoning-parser qwen3" in joined
    assert "--tensor-parallel-size 2" in joined


def test_split_gpu_concedes_policy_to_one_gpu() -> None:
    cfg = load_eval_config("decision_grade").model_copy(update={"serving": ServingPattern.SPLIT_GPU})
    plan = plan_deployment(cfg, policy_port=8100, judge_port=8101)
    assert plan.pattern is ServingPattern.SPLIT_GPU
    policy_gpus = set(plan.policy.gpu_ids)
    judge_gpus = set(plan.phases[1].gpu_ids)
    assert policy_gpus and judge_gpus and policy_gpus.isdisjoint(judge_gpus)
    assert plan.policy.tensor_parallel == 1


def test_judge_phase_has_no_reasoning_parser(eval_config: EvalConfig) -> None:
    judge = plan_deployment(eval_config, policy_port=8100, judge_port=8101).phases[1]
    assert judge.reasoning_parser is None  # judges see raw text, not split reasoning


def test_phase_rejects_tp_gpu_mismatch() -> None:
    with pytest.raises(ValueError, match="tensor_parallel"):
        ServePhase(
            role="policy", model=eval_config_fixture_model(), gpu_ids=(0, 1),
            tensor_parallel=1, max_model_len=4096, gpu_memory_utilization=0.85, port=8100,
        )


def eval_config_fixture_model():
    from medrl.core.config import ModelConfig

    return ModelConfig(hf_id="Qwen/Qwen3.5-9B", text_only=True)


def test_unknown_preset_raises_readably() -> None:
    with pytest.raises(FileNotFoundError, match="preset named"):
        load_eval_config("no_such_preset")


def test_unknown_benchmark_raises_readably(tmp_path) -> None:
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("cluster: local_2xh100\nbenchmarks: [medqa, nonsense]\n")
    with pytest.raises(ValueError, match="nonsense"):
        load_eval_config(cfg_file)


def test_fr_preset_is_french_only() -> None:
    cfg = load_eval_config("fr")
    langs = {TASKS.get(b.name).language.value for b in cfg.benchmarks}
    assert langs == {"fr"}


def test_mcqa_prompt_carries_the_contract() -> None:
    prompt = build_mcqa_user("What is the dose?", ["1 mg", "2 mg", "5 mg", "10 mg"])
    assert prompt.rstrip().endswith("End with 'Answer: <LETTER>'.")
    assert "A. 1 mg" in prompt and "D. 10 mg" in prompt
    # The grammar the guided decoder enforces matches the contract sentence.
    assert MCQA_GRAMMAR == "Answer: [A-E]"


def test_every_decision_benchmark_has_a_verifier() -> None:
    from medrl.core.config import Grade

    for name in TASKS:
        spec = TASKS.get(name)
        if spec.grade is not Grade.DECISION:
            continue
        if spec.verify_style == "letter":
            assert spec.guided_decoding, f"{name}: MCQA without guided decoding"
        if spec.requires_judge:
            assert spec.verify_style == "rubric", f"{name}: judge-requiring but not rubric"


def test_render_is_executable_shell_ordering(eval_config: EvalConfig) -> None:
    text = plan_deployment(eval_config, policy_port=8100, judge_port=8101).render()
    assert text.index("phase 1: policy") < text.index("phase 2: judge")
    assert "vllm serve" in text
