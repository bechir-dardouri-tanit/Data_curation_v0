"""Deployment planning and config loading -- the CPU-testable eval skeleton."""

from __future__ import annotations

import pytest

from medrl.core.config import ClusterConfig, EvalConfig, ServingPattern
from medrl.eval.config_loader import load_eval_config
from medrl.eval.extraction import ExtractionPath, extract_mcqa
from medrl.eval.serving.vllm import ServePhase, plan_deployment
from medrl.eval.tasks.benchmarks import TASKS
from medrl.eval.tasks.prompts import MCQA_GRAMMAR, build_mcqa_user
from medrl.eval.verifiers import verify_letter


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


def test_policy_command_includes_trust_remote_code_when_set(eval_config: EvalConfig) -> None:
    """Models requiring trust_remote_code (e.g., BioMistral) pass the flag to vLLM."""
    from medrl.core.config import ModelConfig

    cfg = eval_config.model_copy(
        update={"model": ModelConfig(hf_id="BioMistral/BioMistral-7B", text_only=True, trust_remote_code=True)}
    )
    argv = plan_deployment(cfg, policy_port=8100).policy.command()
    assert "--trust-remote-code" in argv


def test_policy_command_omits_trust_remote_code_when_not_set(eval_config: EvalConfig) -> None:
    """Models without trust_remote_code do not pass the flag."""
    argv = plan_deployment(eval_config, policy_port=8100).policy.command()
    assert "--trust-remote-code" not in argv


def test_split_gpu_concedes_policy_to_one_gpu() -> None:
    cfg = load_eval_config("decision_grade").model_copy(update={"serving": ServingPattern.SPLIT_GPU})
    plan = plan_deployment(cfg, policy_port=8100, judge_port=8101)
    assert plan.pattern is ServingPattern.SPLIT_GPU
    policy_gpus = set(plan.policy.gpu_ids)
    judge_gpus = set(plan.phases[1].gpu_ids)
    assert policy_gpus and judge_gpus and policy_gpus.isdisjoint(judge_gpus)
    assert plan.policy.tensor_parallel == 1


def test_claim_port_keeps_a_free_planned_port(eval_config: EvalConfig) -> None:
    from medrl.eval.serving.vllm import claim_port, free_port

    phase = plan_deployment(eval_config, policy_port=None, judge_port=None).policy
    drawn = free_port()
    claimed = claim_port(phase.model_copy(update={"port": drawn}))
    assert claimed.port == drawn


def test_claim_port_redraws_a_taken_port(eval_config: EvalConfig) -> None:
    import socket

    from medrl.eval.serving.vllm import claim_port

    phase = plan_deployment(eval_config, policy_port=None, judge_port=None).policy
    with socket.socket() as squatter:
        squatter.bind(("127.0.0.1", 0))
        squatter.listen(1)
        taken = squatter.getsockname()[1]
        claimed = claim_port(phase.model_copy(update={"port": taken}))
    assert claimed.port != taken  # EADDRINUSE after the generation hours, not


def test_judge_phase_parses_reasoning_out_of_content(eval_config: EvalConfig) -> None:
    judge = plan_deployment(eval_config, policy_port=8100, judge_port=8101).phases[1]
    # Insurance for the request-side enable_thinking=False: a leaked <think>
    # block must be split out of content, never ahead of the JSON verdict.
    assert judge.reasoning_parser == "qwen3"


def test_partial_tp_cluster_plans_first_tp_gpus() -> None:
    # Regression: node_8xh100 ships tensor_parallel=4 of 8 GPUs; the planner used to
    # hand the policy all 8 and crash the tp-vs-gpu-count validation.
    cfg = load_eval_config("decision_grade").model_copy(
        update={"cluster": ClusterConfig(name="node_8xh100", num_gpus=8, tensor_parallel=4)}
    )
    plan = plan_deployment(cfg, policy_port=8100, judge_port=8101)
    assert plan.policy.gpu_ids == (0, 1, 2, 3)
    assert plan.policy.tensor_parallel == 4


def test_split_gpu_needs_two_gpus() -> None:
    cfg = load_eval_config("fast").model_copy(
        update={
            "cluster": ClusterConfig(name="one_gpu", num_gpus=1, tensor_parallel=1),
            "serving": ServingPattern.SPLIT_GPU,
        }
    )
    with pytest.raises(ValueError, match="split_gpu"):
        plan_deployment(cfg)


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
    assert prompt.rstrip().endswith("Your final answer:")
    assert "A. 1 mg" in prompt and "D. 10 mg" in prompt
    # The grammar the guided decoder enforces matches the contract sentence.
    assert MCQA_GRAMMAR == "Answer: [A-E]"


def test_mmlu_pro_tasks_use_the_ten_option_alphabet() -> None:
    # Regression: MMLU-Pro has 10 options, but the tasks were constrained AND scored on
    # A-E -- items with gold F-J were forced into a wrong letter by guided decoding and
    # unextractable anyway. ~Half of both benchmarks was silently unwinnable.
    for name in ("mmlu_pro", "mmlu_pro_health"):
        spec = TASKS.get(name)
        assert spec.letters == "ABCDEFGHIJ", name
        assert spec.guided_decoding == "Answer: [A-J]", name
        r = extract_mcqa("Reasoning...\nAnswer: H", spec.letters)
        assert (r.value, r.path) == ("H", ExtractionPath.CONTRACT), name
        assert verify_letter("h", "H", spec.letters), name
    # The A-E default still rejects F-J: five-option tasks keep the tight contract.
    assert extract_mcqa("Answer: H").path is ExtractionPath.FAILED
    assert not verify_letter("H", "H")


def test_task_spec_rejects_gappy_alphabets() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        TASKS.get("medqa").with_overrides(letters="ABDE")


def test_build_mcqa_user_honors_letter_start() -> None:
    # Regression: the parameter was accepted and ignored.
    rendered = build_mcqa_user("Q?", ["first", "second", "third"], letter_start="C")
    assert "C. first" in rendered and "D. second" in rendered and "E. third" in rendered
    with pytest.raises(ValueError, match="one uppercase letter"):
        build_mcqa_user("Q?", ["x"], letter_start="1")


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


def test_env_puts_interpreters_bin_dir_on_path(eval_config: EvalConfig) -> None:
    # Regression: `vllm` is executed bare, so it resolves through the child's PATH.
    # A medrl launched from a non-activated venv (uv run, absolute path) must still
    # find its own vllm -- and sys.executable must NOT be resolved, because uv
    # interpreters are symlinks into /usr/bin and resolving escapes the venv.
    import os
    import sys
    from pathlib import Path

    policy = plan_deployment(eval_config, policy_port=8100).policy
    env = policy.env()
    bin_dir = Path(sys.executable).parent
    if (bin_dir / "vllm").exists():  # vllm installed next to the test interpreter
        assert env["PATH"].split(os.pathsep)[0] == str(bin_dir)
    else:  # CPU CI without vllm: nothing to prepend, placement still present
        assert "PATH" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == ",".join(map(str, policy.gpu_ids))
