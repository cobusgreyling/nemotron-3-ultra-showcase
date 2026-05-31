# NVIDIA Nemotron 3 Ultra — Feature Showcase

A Gradio app that drives the live **NVIDIA Nemotron 3 Ultra** endpoint
(`NVIDIA-Nemotron-3-Ultra-550B-A55B`) to demonstrate its headline features, styled
in NVIDIA green.

![Reasoning playground](assets/01-reasoning-playground.png)

## What it shows

- **Reasoning modes** — `Reasoning ON` (`enable_thinking: true`), `Reasoning OFF`, and
  `Low Effort` (`low_effort: true`), switched live via `chat_template_kwargs`.
- **Reasoning budget** — a slider that caps thinking tokens (`reasoning_budget`).
- **Streaming** — reasoning tokens stream in green, the answer renders alongside.
- **Tool calling** — streaming `delta.tool_calls`, local execution, then a final answer.
- **Model card** — architecture, context length, and recommended generation settings.

| Tab | Feature |
|---|---|
| Reasoning Playground | Modes, reasoning budget, streaming, sample prompts |
| Tool Calling | Streaming tool calls + local tool execution loop |
| Model Card | Architecture summary and recommended settings |

## Benchmarks

How Nemotron 3 Ultra (550B) compares against larger frontier models.

![Intelligence vs output speed](assets/04-intelligence-vs-speed.jpg)

![Benchmark comparison](assets/05-benchmark-comparison.jpg)

![Cost efficiency frontier](assets/06-cost-efficiency-frontier.jpg)

## Run it

Requires Python 3.10+ and an NVIDIA API key with access to the Nemotron 3 Ultra
private endpoint.

```bash
pip install -r requirements.txt
export NVIDIA_API_KEY="nvapi-..."   # never commit this
python app.py
```

Then open http://localhost:7860.

## Configuration

| Setting | Value |
|---|---|
| Base URL | `https://integrate.api.nvidia.com/v1` |
| Model | `private/nvidia/nemotron-3-ultra-550b-a55b` |
| Temperature / top_p | `1.0` / `0.95` (recommended for all modes) |

The endpoint is OpenAI-compatible, so the app uses the `openai` Python client.

## Notes

- With thinking on, the model may emit empty `<think></think>` tags — expected when it
  decides it does not need to reason to answer confidently.
- Reasoning mode is verbose; `max_tokens` is generous and adjustable in the UI.
- The bundled tool is a safe, whitelisted arithmetic evaluator used only to demonstrate
  the tool-call loop.

See [`BLOG.md`](BLOG.md) for the write-up.
