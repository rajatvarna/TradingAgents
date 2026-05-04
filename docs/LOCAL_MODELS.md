# Local Model Servers: LM-Studio, Llama.cpp & Ollama

This guide explains how to set up and use local LLM servers with TradingAgents.

## Overview

TradingAgents now supports three local model server platforms:

| Platform | Port | Type | Setup Complexity |
|----------|------|------|------------------|
| **LM-Studio** | 8000 | UI + API | Easy (GUI) |
| **Llama.cpp** | 8001 | API only | Medium (CLI) |
| **Ollama** | 11434 | API + Models | Easy (CLI + auto-download) |

All three use the OpenAI-compatible `/v1/chat/completions` API, so integration is seamless.

---

## 1. LM-Studio

**LM-Studio** is a desktop application with a graphical interface for downloading, managing, and running local LLMs.

### Installation

1. **Download LM-Studio**
   - Visit: https://lmstudio.ai/
   - Download for your OS (Windows, macOS, Linux)
   - Install the application

2. **Launch LM-Studio**
   - Open the application
   - Go to the **Models** tab
   - Search for and download a model (e.g., `Qwen3:latest`, `Mistral`, `Llama3`)
   - Wait for download to complete

3. **Start the Server**
   - Go to the **Local Server** tab
   - Select your downloaded model from the dropdown
   - Click **Start Server**
   - You should see: `Server running on http://localhost:8000/v1`

### Using with TradingAgents

```bash
# Run TradingAgents CLI
python -m cli.main

# Select provider: "LM-Studio"
# Enter model ID: (e.g., "qwen3-32b-instruct", "mistral-7b")
# Default port is 8000, no API key needed
```

### Custom Port

If LM-Studio is running on a different port:

```bash
# Set in .env
LM_STUDIO_BASE_URL=http://localhost:9000/v1
```

Or in the CLI when prompted, enter the custom URL.

### Model Recommendations

**For Quick Thinking (Fast Tasks):**
- Qwen3:latest (8B) - fastest, good quality
- Mistral-7B - balanced
- Phi-4 (3.8B) - ultra-fast

**For Deep Thinking (Complex Analysis):**
- Qwen3.5 (32B) - best balance
- Llama3.1 (70B) - most capable, slower
- Mixtral-8x7B - good reasoning

---

## 2. Llama.cpp

**Llama.cpp** is a lightweight C++ inference engine for GGUF-format models. It's faster than LM-Studio for local inference.

### Installation

1. **Download Llama.cpp**
   - Visit: https://github.com/ggerganov/llama.cpp/releases
   - Download the latest release for your OS
   - Extract the archive

2. **Download a Model (GGUF format)**
   ```bash
   # Example: Download Qwen2.5-7B in GGUF format
   cd llama.cpp
   # Download from HuggingFace (e.g., ggml-org/models)
   ```

3. **Start the Server**
   ```bash
   # Default: port 8001
   ./server -m path/to/model.gguf -n 2048

   # Or with GPU acceleration (CUDA)
   ./server -m path/to/model.gguf -n 2048 -ngl 33

   # Custom port
   ./server -m path/to/model.gguf --port 9001
   ```

   You should see: `Server listening at http://0.0.0.0:8001`

### Using with TradingAgents

```bash
# Run TradingAgents CLI
python -m cli.main

# Select provider: "Llama.cpp"
# Enter model ID: (e.g., "qwen2.5-7b", "mistral-7b")
# Default port is 8001, no API key needed
```

### Custom Port

If Llama.cpp is running on a different port:

```bash
# Set in .env
LLAMA_CPP_BASE_URL=http://localhost:9001/v1
```

### GPU Acceleration

**CUDA (NVIDIA)**
```bash
./server -m path/to/model.gguf -ngl 33 -n 2048
```

**Metal (macOS)**
```bash
./server -m path/to/model.gguf -ngl 1 -n 2048
```

**ROCm (AMD)**
```bash
./server -m path/to/model.gguf -ngl 33 -n 2048
```

### Model Recommendations

Popular GGUF models:
- `Qwen/Qwen2.5-7B-Instruct-GGUF`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `meta-llama/Llama-3.2-1B-Instruct`
- `TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF`

---

## 3. Ollama

**Ollama** is the easiest option - it automatically downloads and runs models.

### Installation

1. **Download Ollama**
   - Visit: https://ollama.ai/
   - Download for your OS (Windows, macOS, Linux)
   - Install

2. **Start Ollama**
   - On macOS/Linux: `ollama serve`
   - On Windows: Ollama starts automatically as a service
   - Should run on `http://localhost:11434`

3. **Download a Model**
   ```bash
   ollama pull qwen3:latest
   ollama pull mistral:latest
   ollama pull llama3:latest
   ```

4. **Run a Model**
   ```bash
   # The server is already running, just specify the model in TradingAgents
   # OR run specific model:
   ollama run qwen3:latest
   ```

### Using with TradingAgents

```bash
# Run TradingAgents CLI
python -m cli.main

# Select provider: "Ollama"
# Select model: "qwen3:latest", "mistral:latest", etc.
# No API key needed
```

### Available Models

```bash
ollama ls  # List installed models
ollama pull <model>  # Download model
```

Popular models:
- `qwen3:latest` (8B, fast, good quality)
- `mistral:latest` (7B, fast)
- `llama3:latest` (8B, good reasoning)
- `gpt-oss:latest` (20B, higher quality)
- `glm-4.7-flash:latest` (30B, best reasoning)

---

## Comparison

| Feature | LM-Studio | Llama.cpp | Ollama |
|---------|-----------|-----------|--------|
| **Setup** | GUI (easiest) | CLI (medium) | CLI (easiest) |
| **Model Management** | Download in UI | Manual download | Auto-download |
| **Speed** | Good | Excellent | Good |
| **GPU Support** | ✓ | ✓ | ✓ |
| **Port** | 8000 | 8001 | 11434 |
| **API Key** | None | None | None |
| **Models Available** | Varies | Varies | Large library |
| **Recommended for** | Desktop users | Performance users | Everyone |

---

## Configuration Files

### .env Variables

```bash
# Override default ports if using custom ones
LM_STUDIO_BASE_URL=http://localhost:8000/v1
LLAMA_CPP_BASE_URL=http://localhost:8001/v1
OLLAMA_BASE_URL=http://localhost:11434/v1
```

### Trading Agents Config

Set in CLI or update `default_config.py`:

```python
DEFAULT_CONFIG = {
    "llm_provider": "ollama",  # or "lm-studio", "llama-cpp"
    "deep_think_llm": "qwen3:latest",
    "quick_think_llm": "mistral:latest",
    "backend_url": None,  # Uses default port, override if needed
}
```

---

## Troubleshooting

### Port Already in Use

If the default port is occupied:

**LM-Studio:**
- Go to Settings → Server → change port

**Llama.cpp:**
```bash
./server -m model.gguf --port 9001
# Then set LLAMA_CPP_BASE_URL=http://localhost:9001/v1
```

**Ollama:**
- Restart: `ollama serve --port 9001`

### Model Not Found

**LM-Studio:**
- Go to Models tab → search and download the model again

**Llama.cpp:**
- Download the .gguf file manually from HuggingFace

**Ollama:**
```bash
ollama pull model-name
```

### Server Not Responding

**LM-Studio:**
- Ensure server is running (check Local Server tab)

**Llama.cpp:**
- Run: `./server -m model.gguf -n 2048`

**Ollama:**
```bash
ollama serve  # On macOS/Linux
# On Windows, ensure Ollama service is running
```

### Memory Issues

Reduce context length or model size:

```bash
# Llama.cpp: reduce context (-n parameter)
./server -m model.gguf -n 1024

# Ollama: use smaller model
ollama pull mistral:latest  # 7B instead of 70B
```

---

## Performance Tips

1. **Use GPU acceleration** for 2x-5x speedup
2. **Start with smaller models** (7B-13B) for fast iteration
3. **Increase batch size** in server config for better throughput
4. **Use quantized models** (GGUF in llama.cpp) for speed

---

## Next Steps

- Start the local server of your choice
- Run: `python -m cli.main`
- Select your local provider and model
- Begin analysis!

For model recommendations specific to financial analysis and trading decisions, see the main README.
