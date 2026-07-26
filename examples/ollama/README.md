# Custom Ollama Modelfiles for TradingAgents

These example [Modelfiles](https://docs.ollama.com/modelfile) pre-tune a
local Qwen3 model for TradingAgents' analyst/trader prompts, so you don't
have to pass the same `num_ctx`/`temperature` overrides on every run. See
[`docs/LOCAL_MODELS.md`](../../docs/LOCAL_MODELS.md) for general Ollama
setup with TradingAgents.

## Profiles

| Profile | Base tag | Context | Quantization | Best for |
|---|---|---|---|---|
| `trading-fast` | `qwen3:8b-q4_K_M` | 8192 tokens | Q4_K_M | Quick iteration, the shallow-thinker analyst passes, lower VRAM |
| `trading-accurate` | `qwen3:8b-q8_0` | 4096 tokens | Q8_0 | Higher-fidelity deep-thinker passes when VRAM allows |

Both profiles pin `temperature 0.2` and `top_p 0.9` — low-variance settings
appropriate for financial analysis, where you want reproducible reasoning
rather than creative variation.

## Building a profile

```bash
cd examples/ollama
ollama create trading-fast -f Modelfile.trading-fast
ollama create trading-accurate -f Modelfile.trading-accurate
```

## Using a profile

From the CLI, select "Ollama" as the provider and enter the profile name
(`trading-fast` or `trading-accurate`) as the model:

```bash
python -m cli.main
# Select provider: "Ollama"
# Select model: "trading-fast"
```

Or set it directly via the non-interactive env override:

```bash
export TRADINGAGENTS_LLM_PROVIDER=ollama
export TRADINGAGENTS_QUICK_THINK_LLM=trading-fast
export TRADINGAGENTS_DEEP_THINK_LLM=trading-accurate
```

## Tuning for your hardware

- **Lower VRAM:** drop to a smaller base model (e.g. `qwen3:4b`) or a
  lighter quantization (`q4_0` instead of `q4_K_M`/`q8_0`).
- **Larger context needs** (e.g. long debate histories with many rounds):
  raise `num_ctx`, but expect proportionally higher memory use and latency.
- **More deterministic output:** lower `temperature` further (e.g. `0.0`–`0.1`).

Run `ollama pull qwen3:8b-q4_K_M` and `ollama pull qwen3:8b-q8_0` first if
you don't already have those base images — `ollama create` builds each
custom tag from its matching local base image (the exact `FROM` line in
the Modelfile), it doesn't download one.
