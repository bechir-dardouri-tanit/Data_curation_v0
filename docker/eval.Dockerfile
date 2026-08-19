# Serving/eval image: vLLM runtime for the 2xH100 host.
# Pinned by digest at build time in production; tag kept explicit here.
FROM vllm/vllm-openai:v0.11.0
ENV HF_HOME=/scratch/medrl/hf \
    MEDRL_SCRATCH=/scratch/medrl \
    VLLM_USE_V1=1
WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[eval]"
