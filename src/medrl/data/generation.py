"""Teacher generation and distillation for the medrl pipeline.

Handles generation of synthetic training data using teacher models, including
response generation for SFT, preference pairs for RLHF, and rationale extraction
for chain-of-thought distillation.

Typical usage:
    >>> from medrl.data.generation import TeacherGenerator
    >>> generator = TeacherGenerator(model="gpt-4")
    >>> responses = generator.generate(prompts, max_tokens=512)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from medrl.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Generation Configuration
# =============================================================================


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for teacher model generation.

    Attributes:
        model: Model identifier (OpenAI, Anthropic, or local).
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens to generate.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling threshold.
        presence_penalty: Presence penalty (-2.0 to 2.0).
        frequency_penalty: Frequency penalty (-2.0 to 2.0).
        num_samples: Number of samples per prompt.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.
    """

    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    num_samples: int = 1
    timeout: int = 60
    max_retries: int = 3


@dataclass(frozen=True)
class PromptTemplate:
    """A prompt template for generation.

    Attributes:
        template: Template string with {variable} placeholders.
        variables: List of required variable names.
        system_message: Optional system message.
        default_values: Default values for variables.

    Examples:
        >>> template = PromptTemplate(
        ...     template="Question: {question}\\nAnswer:",
        ...     variables=["question"]
        ... )
        >>> template.render(question="What is NSAID?")
        'Question: What is NSAID?\\nAnswer:'
    """

    template: str
    variables: tuple[str, ...] = ()
    system_message: str = ""
    default_values: dict[str, str] = field(default_factory=dict)

    def render(self, **kwargs: str) -> str:
        """Render the template with provided values.

        Args:
            **kwargs: Variable values.

        Returns:
            Rendered prompt string.
        """
        values = self.default_values | kwargs
        return self.template.format(**values)

    def validate(self, **kwargs: str) -> bool:
        """Check if all required variables are provided.

        Args:
            **kwargs: Variable values to check.

        Returns:
            True if all required variables are present.
        """
        missing = set(self.variables) - set(kwargs) - set(self.default_values)
        return not missing


# =============================================================================
# Teacher Generator
# =============================================================================


class TeacherGenerator:
    """Generate synthetic training data using teacher models.

    Supports OpenAI, Anthropic, and local models via a unified interface. Handles
    batching, retries, and rate limiting.

    Attributes:
        config: Generation configuration.
        client: Underlying API client.

    Examples:
        >>> generator = TeacherGenerator(model="gpt-4o")
        >>> prompts = ["What is aspirin?", "What is ibuprofen?"]
        >>> responses = generator.generate_batch(prompts)
    """

    def __init__(self, config: GenerationConfig | None = None) -> None:
        """Initialize the teacher generator.

        Args:
            config: Generation configuration.
        """
        self.config = config or GenerationConfig()
        self._client: Any = None
        self._provider: str = self._detect_provider(self.config.model)

    def _detect_provider(self, model: str) -> str:
        """Detect the provider from model name.

        Args:
            model: Model identifier.

        Returns:
            Provider name ("openai", "anthropic", "local").
        """
        if model.startswith("gpt-"):
            return "openai"
        elif model.startswith("claude-"):
            return "anthropic"
        else:
            return "local"

    @property
    def client(self) -> Any:
        """Get or create the API client.

        Returns:
            API client instance.
        """
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> Any:
        """Create the appropriate API client.

        Returns:
            Client instance.

        Raises:
            ImportError: If required package is not installed.
        """
        if self._provider == "openai":
            try:
                from openai import OpenAI
                return OpenAI(timeout=self.config.timeout)
            except ImportError as e:
                raise ImportError(
                    "openai is required for OpenAI models. "
                    "Install with: pip install openai"
                ) from e

        elif self._provider == "anthropic":
            try:
                from anthropic import Anthropic
                return Anthropic(timeout=self.config.timeout)
            except ImportError as e:
                raise ImportError(
                    "anthropic is required for Claude models. "
                    "Install with: pip install anthropic"
                ) from e

        else:
            raise ValueError(f"Local models not yet supported: {self.config.model}")

    def generate(
        self,
        prompt: str,
        system_message: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a single response.

        Args:
            prompt: Input prompt.
            system_message: Optional system message.
            **kwargs: Additional generation parameters (override config).

        Returns:
            Generated response text.

        Raises:
            RuntimeError: If generation fails after retries.
        """
        params = self._merge_params(kwargs)

        if self._provider == "openai":
            return self._generate_openai(prompt, system_message, params)
        elif self._provider == "anthropic":
            return self._generate_anthropic(prompt, system_message, params)
        else:
            raise RuntimeError(f"Unsupported provider: {self._provider}")

    def _generate_openai(
        self,
        prompt: str,
        system_message: str | None,
        params: dict[str, Any],
    ) -> str:
        """Generate using OpenAI API.

        Args:
            prompt: Input prompt.
            system_message: Optional system message.
            params: Generation parameters.

        Returns:
            Generated response.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=params.get("temperature", self.config.temperature),
                    max_tokens=params.get("max_tokens", self.config.max_tokens),
                    top_p=params.get("top_p", self.config.top_p),
                    presence_penalty=params.get(
                        "presence_penalty", self.config.presence_penalty
                    ),
                    frequency_penalty=params.get(
                        "frequency_penalty", self.config.frequency_penalty
                    ),
                )
                return response.choices[0].message.content or ""

            except Exception as e:
                logger.warning(f"OpenAI generation attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise RuntimeError(f"OpenAI generation failed after {self.config.max_retries} attempts") from e

        return ""  # Should never reach here

    def _generate_anthropic(
        self,
        prompt: str,
        system_message: str | None,
        params: dict[str, Any],
    ) -> str:
        """Generate using Anthropic API.

        Args:
            prompt: Input prompt.
            system_message: Optional system message.
            params: Generation parameters.

        Returns:
            Generated response.
        """
        messages = [{"role": "user", "content": prompt}]

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.messages.create(
                    model=self.config.model,
                    system=system_message or "",
                    messages=messages,
                    temperature=params.get("temperature", self.config.temperature),
                    max_tokens=params.get("max_tokens", self.config.max_tokens),
                    top_p=params.get("top_p", self.config.top_p),
                )
                return response.content[0].text  # type: ignore[no-any-return]

            except Exception as e:
                logger.warning(f"Anthropic generation attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise RuntimeError(f"Anthropic generation failed after {self.config.max_retries} attempts") from e

        return ""  # Should never reach here

    def generate_batch(
        self,
        prompts: Sequence[str],
        system_message: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Generate responses for multiple prompts.

        Args:
            prompts: Input prompts.
            system_message: Optional system message.
            **kwargs: Additional generation parameters.

        Returns:
            List of generated responses.
        """
        responses = []
        for prompt in prompts:
            try:
                response = self.generate(prompt, system_message, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to generate response for prompt: {prompt[:50]}... Error: {e}")
                responses.append("")  # Empty string for failed generations

        return responses

    def _merge_params(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge config with runtime parameters.

        Args:
            kwargs: Runtime parameters.

        Returns:
            Merged parameters.
        """
        base = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "presence_penalty": self.config.presence_penalty,
            "frequency_penalty": self.config.frequency_penalty,
        }
        return base | kwargs


# =============================================================================
# SFT Data Generation
# =============================================================================


def generate_sft_data(
    prompts: Sequence[str],
    generator: TeacherGenerator,
    system_message: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate SFT training data from prompts.

    Args:
        prompts: Input prompts.
        generator: Teacher generator instance.
        system_message: Optional system message.
        **kwargs: Additional generation parameters.

    Returns:
        List of SFT items with prompt and response.

    Examples:
        >>> generator = TeacherGenerator(model="gpt-4o")
        >>> prompts = ["What is aspirin?"]
        >>> sft_items = generate_sft_data(prompts, generator)
        >>> sft_items[0]["response"]
        'Aspirin is a non-steroidal anti-inflammatory drug...'
    """
    responses = generator.generate_batch(prompts, system_message, **kwargs)

    sft_items = []
    for prompt, response in zip(prompts, responses, strict=False):
        if response:  # Skip failed generations
            sft_items.append({"prompt": prompt, "response": response})

    logger.info(f"Generated {len(sft_items)} SFT items from {len(prompts)} prompts")

    return sft_items


# =============================================================================
# Preference Pair Generation
# =============================================================================


def generate_preference_pairs(
    prompts: Sequence[str],
    generator: TeacherGenerator,
    system_message: str | None = None,
    temperature_low: float = 0.3,
    temperature_high: float = 1.0,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate preference pairs for RLHF/DPO training.

    Generates two responses per prompt: one low-temperature (high quality)
    and one high-temperature (more diverse). The low-temp response is used
    as the "chosen" response and the high-temp as "rejected" (or vice versa
    depending on quality).

    Args:
        prompts: Input prompts.
        generator: Teacher generator instance.
        system_message: Optional system message.
        temperature_low: Temperature for chosen response.
        temperature_high: Temperature for rejected response.
        **kwargs: Additional generation parameters.

    Returns:
        List of preference pairs.

    Examples:
        >>> generator = TeacherGenerator(model="gpt-4o")
        >>> pairs = generate_preference_pairs(
        ...     ["What is aspirin?"],
        ...     generator,
        ...     temperature_low=0.3,
        ...     temperature_high=1.0
        ... )
        >>> pairs[0]["chosen"]
        'Aspirin is a medication that reduces pain...'
    """
    # Generate chosen (high quality)
    chosen_responses = generator.generate_batch(
        prompts, system_message, temperature=temperature_low, **kwargs
    )

    # Generate rejected (more diverse)
    rejected_responses = generator.generate_batch(
        prompts, system_message, temperature=temperature_high, **kwargs
    )

    pairs = []
    for prompt, chosen, rejected in zip(prompts, chosen_responses, rejected_responses, strict=False):
        if chosen and rejected:
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    logger.info(f"Generated {len(pairs)} preference pairs from {len(prompts)} prompts")

    return pairs


# =============================================================================
# Rationale Extraction (CoT Distillation)
# =============================================================================


def extract_rationales(
    items: Sequence[dict[str, Any]],
    generator: TeacherGenerator,
    prompt_template: str = "Think step by step and explain your reasoning for: {question}",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Extract reasoning rationales from a teacher model.

    Used for chain-of-thought distillation: the teacher provides reasoning
    steps that the student model learns to replicate.

    Args:
        items: Items with questions/problems.
        generator: Teacher generator instance.
        prompt_template: Template for rationale extraction.
        **kwargs: Additional generation parameters.

    Returns:
        Items with added rationales.

    Examples:
        >>> items = [{"question": "What is 2+2?"}]
        >>> generator = TeacherGenerator(model="gpt-4o")
        >>> with_rationale = extract_rationales(items, generator)
        >>> "rationale" in with_rationale[0]
        True
    """
    enriched_items = []

    for item in items:
        question = item.get("question", item.get("prompt", ""))
        prompt = prompt_template.format(question=question)

        try:
            rationale = generator.generate(prompt, **kwargs)
            enriched_item = item.copy()
            enriched_item["rationale"] = rationale
            enriched_items.append(enriched_item)
        except Exception as e:
            logger.error(f"Failed to extract rationale for item: {e}")
            enriched_items.append(item)  # Keep original without rationale

    logger.info(f"Extracted rationales for {len(enriched_items)} items")

    return enriched_items


# =============================================================================
# Data Augmentation
# =============================================================================


def augment_dataset(
    items: Sequence[dict[str, Any]],
    generator: TeacherGenerator,
    instruction_template: str = "Rewrite the following question in a different way: {question}",
    n_variants: int = 2,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Augment dataset by generating paraphrased variants.

    Args:
        items: Original items.
        generator: Teacher generator instance.
        instruction_template: Template for paraphrase instruction.
        n_variants: Number of variants per item.
        **kwargs: Additional generation parameters.

    Returns:
        Original items plus augmented variants.

    Examples:
        >>> items = [{"question": "What is aspirin?"}]
        >>> generator = TeacherGenerator(model="gpt-4o")
        >>> augmented = augment_dataset(items, generator, n_variants=1)
        >>> len(augmented)
        2
    """
    augmented = list(items)  # Start with originals

    for item in items:
        question = item.get("question", item.get("prompt", ""))

        for i in range(n_variants):
            prompt = instruction_template.format(question=question)
            try:
                paraphrase = generator.generate(prompt, temperature=0.8, **kwargs)

                new_item = item.copy()
                new_item["question"] = paraphrase
                new_item["original_question"] = question
                new_item["augmentation_id"] = i
                augmented.append(new_item)

            except Exception as e:
                logger.error(f"Failed to generate paraphrase: {e}")

    logger.info(f"Augmented {len(items)} items to {len(augmented)} total")

    return augmented
