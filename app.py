"""
NVIDIA Nemotron 3 Ultra — Feature Showcase
A Gradio app demonstrating the headline capabilities of
NVIDIA-Nemotron-3-Ultra-550B-A55B:

  - Reasoning modes: ON / OFF / Low Effort  (chat_template_kwargs)
  - reasoning_budget  (cap the thinking tokens)
  - Streaming reasoning + answer
  - Streaming tool calls (delta.tool_calls)

Set your key first:  export NVIDIA_API_KEY="nvapi-..."
Then:                python app.py
"""

import json
import os
import re

import gradio as gr
from openai import OpenAI

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "private/nvidia/nemotron-3-ultra-550b-a55b"
STREAM_TIMEOUT_SECONDS = 1800

NVIDIA_GREEN = "#76B900"


def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise gr.Error(
            "NVIDIA_API_KEY is not set. Run  export NVIDIA_API_KEY=\"nvapi-...\"  "
            "before launching the app."
        )
    return OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=api_key,
        default_headers={"NVCF-POLL-SECONDS": "1800"},
    )


def build_extra_body(mode: str, budget: int | None) -> dict:
    """Translate the UI mode into Nemotron chat_template_kwargs."""
    kwargs: dict = {}
    if mode == "Reasoning ON":
        kwargs["enable_thinking"] = True
        if budget and budget > 0:
            kwargs["reasoning_budget"] = int(budget)
    elif mode == "Low Effort":
        kwargs["enable_thinking"] = True
        kwargs["low_effort"] = True
    else:  # Reasoning OFF
        kwargs["enable_thinking"] = False
    return {"chat_template_kwargs": kwargs}


# --------------------------------------------------------------------------- #
# Tab 1 — Reasoning Playground
# --------------------------------------------------------------------------- #
def run_reasoning(prompt, system_prompt, mode, budget, max_tokens):
    if not prompt or not prompt.strip():
        yield "", "_Enter a prompt to begin._", ""
        return

    client = get_client()
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": prompt.strip()})

    extra_body = build_extra_body(mode, budget)
    meta = (
        f"`model={MODEL}`  ·  `mode={mode}`  ·  "
        f"`reasoning_budget={extra_body['chat_template_kwargs'].get('reasoning_budget', '—')}`  ·  "
        f"`max_tokens={int(max_tokens)}`"
    )

    reasoning, answer = "", ""
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=1.0,
            top_p=0.95,
            max_tokens=int(max_tokens),
            stream=True,
            timeout=STREAM_TIMEOUT_SECONDS,
            extra_body=extra_body,
        )
        for chunk in stream:
            for choice in chunk.choices:
                delta = choice.delta
                r = getattr(delta, "reasoning", None) or getattr(
                    delta, "reasoning_content", None
                )
                if r:
                    reasoning += r
                    yield reasoning, answer or "_thinking…_", meta
                c = getattr(delta, "content", None)
                if c:
                    answer += c
                    yield reasoning, answer, meta
    except Exception as e:  # noqa: BLE001
        yield reasoning, f"**Error:** {e}", meta
        return

    if not reasoning.strip():
        reasoning = "_(no thinking tokens — the model answered directly)_"
    if not answer.strip():
        answer = "_(empty response)_"
    yield reasoning, answer, meta


# --------------------------------------------------------------------------- #
# Tab 2 — Tool Calling
# --------------------------------------------------------------------------- #
MATH_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "get_math_answer",
        "description": "Returns an exact arithmetic result",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"],
        },
    },
}


def get_math_answer(expression: str) -> str:
    """Safe evaluator for simple arithmetic."""
    expr = re.sub(r"\s+", "", expression)
    if not re.match(r"^[\d+\-*/().]+$", expr):
        return "Error: only numbers and + - * / ( ) allowed"
    try:
        return str(eval(expr))  # noqa: S307 — input is whitelisted above
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


def run_tool_call(prompt):
    if not prompt or not prompt.strip():
        yield "_Enter a prompt to begin._", "", ""
        return

    client = get_client()
    user_msg = {"role": "user", "content": prompt.strip()}

    reasoning, trace = "", ""
    acc: dict = {}
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[user_msg],
            temperature=1.0,
            top_p=0.95,
            max_tokens=8192,
            stream=True,
            timeout=STREAM_TIMEOUT_SECONDS,
            tools=[MATH_TOOL_SPEC],
            tool_choice="auto",
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        for chunk in stream:
            for choice in chunk.choices:
                delta = choice.delta
                r = getattr(delta, "reasoning", None) or getattr(
                    delta, "reasoning_content", None
                )
                if r:
                    reasoning += r
                    yield reasoning, trace, ""
                for tc in getattr(delta, "tool_calls", None) or []:
                    idx = getattr(tc, "index", 0) or 0
                    acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    acc[idx]["id"] = acc[idx]["id"] or (getattr(tc, "id", "") or "")
                    fn = getattr(tc, "function", None)
                    name_part = (getattr(fn, "name", None) or "") if fn else ""
                    args_part = (getattr(fn, "arguments", None) or "") if fn else ""
                    if name_part and not acc[idx]["name"]:
                        # The model may resend the name across deltas; take it once.
                        trace += f"→ tool#{idx} name    {name_part}\n"
                        acc[idx]["name"] = name_part
                    if args_part:
                        trace += f"→ tool#{idx} args+   {args_part}\n"
                        acc[idx]["arguments"] += args_part
                    yield reasoning, trace, ""
    except Exception as e:  # noqa: BLE001
        yield reasoning, trace, f"**Error:** {e}"
        return

    # Execute the requested tools locally, then send results back for a final answer.
    assistant_tool_calls, tool_results = [], []
    for idx in sorted(acc):
        tc = acc[idx]
        if not tc["name"]:
            continue
        try:
            args = json.loads(tc["arguments"]) if tc["arguments"].strip() else {}
        except json.JSONDecodeError:
            args = {}
        result = (
            get_math_answer(args.get("expression", ""))
            if tc["name"] == "get_math_answer"
            else f"(unknown tool: {tc['name']})"
        )
        trace += f"\n⚙  executed  {tc['name']}({args}) → {result}\n"
        assistant_tool_calls.append(
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
        )
        tool_results.append(
            {"role": "tool", "tool_call_id": tc["id"], "content": result}
        )
        yield reasoning, trace, ""

    if not assistant_tool_calls:
        yield reasoning or "_(no thinking tokens)_", trace, "_(model did not call a tool)_"
        return

    follow_up = [
        user_msg,
        {"role": "assistant", "content": None, "tool_calls": assistant_tool_calls},
        *tool_results,
    ]
    final = ""
    final_stream = client.chat.completions.create(
        model=MODEL,
        messages=follow_up,
        temperature=1.0,
        top_p=0.95,
        max_tokens=8192,
        stream=True,
        timeout=STREAM_TIMEOUT_SECONDS,
        tools=[MATH_TOOL_SPEC],
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )
    for chunk in final_stream:
        for choice in chunk.choices:
            c = getattr(choice.delta, "content", None)
            if c:
                final += c
                yield reasoning or "_(no thinking tokens)_", trace, final


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
CSS = f"""
.gradio-container {{ max-width: 1180px !important; }}
#hero {{
    background: linear-gradient(135deg, #0b0b0b 0%, #141a0c 100%);
    border-left: 6px solid {NVIDIA_GREEN};
    border-radius: 10px;
    padding: 26px 30px;
    margin-bottom: 8px;
}}
#hero h1 {{ color: {NVIDIA_GREEN}; margin: 0 0 6px 0; font-size: 30px; letter-spacing: .3px; }}
#hero p  {{ color: #cfcfcf; margin: 0; font-size: 15px; }}
.nv-pill {{
    display:inline-block; background:{NVIDIA_GREEN}; color:#000;
    font-weight:700; font-size:12px; padding:3px 10px; border-radius:20px; margin-right:8px;
}}
.gr-button-primary {{ background:{NVIDIA_GREEN} !important; border-color:{NVIDIA_GREEN} !important; color:#000 !important; font-weight:700 !important; }}
#reasoning_box textarea {{ color:{NVIDIA_GREEN} !important; font-family: ui-monospace, monospace !important; font-size:13px !important; }}
footer {{ visibility: hidden; }}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#f3f9e6", c100="#e4f3c2", c200="#cfe98f", c300="#b6dd58",
        c400="#9ccb2e", c500=NVIDIA_GREEN, c600="#5e9400", c700="#487000",
        c800="#324f00", c900="#1e3000", c950="#0f1a00",
    ),
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(body_background_fill="#0a0a0a", block_background_fill="#141414")

SAMPLE_PROMPTS = {
    "Enterprise architecture": (
        "Design a highly available, multi-region microservices architecture for a "
        "global financial trading platform. The system must process 100,000 "
        "transactions per second with sub-millisecond latency. Address data "
        "replication, failover, and an AI fraud-detection pipeline. Let's think step by step."
    ),
    "Supply-chain optimization": (
        "A plant in Taiwan produces 5,000 components daily. Sea shipping to the US hub "
        "takes 14 days ($2/unit); air takes 2 days ($15/unit). The hub must fulfil a "
        "40,000-component contract in 10 days but holds only 15,000 units. Give a hybrid "
        "shipping plan that meets the deadline at minimum cost, with the exact cost calculation."
    ),
    "Advanced coding": (
        "Write a production-ready FastAPI service that receives Stripe webhooks, verifies "
        "the signature, parses the event, and asynchronously updates PostgreSQL via "
        "SQLAlchemy. Include error handling, logging, and retry logic for deadlocks."
    ),
    "Quick fact (try Low Effort)": "What is NVIDIA?",
}

with gr.Blocks(title="Nemotron 3 Ultra — Showcase") as demo:
    gr.HTML(
        """
        <div id="hero">
          <h1>NVIDIA Nemotron 3 Ultra</h1>
          <p>
            <span class="nv-pill">550B / 55B active</span>
            <span class="nv-pill">Latent MoE · Mamba2-Transformer</span>
            <span class="nv-pill">1M context</span>
            <span class="nv-pill">Reasoning ON / OFF / Low Effort</span>
          </p>
          <p style="margin-top:10px">An open frontier-reasoning model for long-running autonomous agents.
          This app drives the live NVIDIA endpoint to show its switchable reasoning, reasoning budget, streaming, and tool calling.</p>
        </div>
        """
    )

    with gr.Tabs():
        # ----- Reasoning Playground ----- #
        with gr.Tab("Reasoning Playground"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(
                        label="Prompt",
                        lines=6,
                        placeholder="Ask Nemotron 3 Ultra something hard…",
                    )
                    with gr.Accordion("System prompt (optional)", open=False):
                        system_prompt = gr.Textbox(
                            label="System prompt",
                            lines=2,
                            value="You are a senior enterprise solutions architect.",
                        )
                    gr.Markdown("**Try a sample prompt:**")
                    with gr.Row():
                        sample_btns = [gr.Button(name, size="sm") for name in SAMPLE_PROMPTS]
                    run_btn = gr.Button("Run ▸", variant="primary")
                with gr.Column(scale=1):
                    mode = gr.Radio(
                        ["Reasoning ON", "Low Effort", "Reasoning OFF"],
                        value="Reasoning ON",
                        label="Reasoning mode",
                        info="chat_template_kwargs sent to the model",
                    )
                    budget = gr.Slider(
                        0, 16384, value=0, step=512,
                        label="reasoning_budget (0 = unlimited)",
                        info="Caps thinking tokens. Only applies to Reasoning ON.",
                    )
                    max_tokens = gr.Slider(
                        1024, 64000, value=8192, step=1024, label="max_tokens"
                    )
            meta_md = gr.Markdown()
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🧠 Reasoning tokens")
                    reasoning_out = gr.Textbox(
                        label="", lines=14, elem_id="reasoning_box"
                    )
                with gr.Column():
                    gr.Markdown("### ✅ Answer")
                    answer_out = gr.Markdown()

            run_btn.click(
                run_reasoning,
                [prompt, system_prompt, mode, budget, max_tokens],
                [reasoning_out, answer_out, meta_md],
            )
            for name, btn in zip(SAMPLE_PROMPTS, sample_btns):
                btn.click(lambda v=SAMPLE_PROMPTS[name]: v, None, prompt)

        # ----- Tool Calling ----- #
        with gr.Tab("Tool Calling"):
            gr.Markdown(
                "Nemotron 3 Ultra is post-trained for **agent harnesses**. Here it reasons, "
                "emits a streaming `delta.tool_calls`, we execute the tool locally, then it "
                "returns the final answer. The bundled tool is a safe arithmetic evaluator."
            )
            tool_prompt = gr.Textbox(
                label="Prompt",
                lines=2,
                value="Use the get_math_answer tool to compute 123+456. Return only the final answer.",
            )
            tool_btn = gr.Button("Run tool-call demo ▸", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🧠 Reasoning")
                    tool_reasoning = gr.Textbox(
                        label="", lines=8, elem_id="reasoning_box"
                    )
                with gr.Column():
                    gr.Markdown("### 🔧 Tool-call stream")
                    tool_trace = gr.Textbox(label="", lines=8, elem_id="reasoning_box")
            gr.Markdown("### ✅ Final answer")
            tool_answer = gr.Markdown()
            tool_btn.click(
                run_tool_call, [tool_prompt], [tool_reasoning, tool_trace, tool_answer]
            )

        # ----- Model Card ----- #
        with gr.Tab("Model Card"):
            gr.Markdown(
                f"""
### NVIDIA-Nemotron-3-Ultra-550B-A55B

| Attribute | Value |
|---|---|
| Total parameters | 550B |
| Active parameters | 55B |
| Architecture | Latent MoE Hybrid Mamba2-Transformer |
| Context length | Up to 1M tokens |
| Reasoning modes | ON (`enable_thinking: true`) / OFF / Low Effort (`low_effort: true`) |
| Reasoning budget | Optional (`reasoning_budget: <int>`) |
| Recommended max tokens (reasoning ON) | 264K |
| Recommended temperature / top_p | 1.0 / 0.95 |

**Three breakthroughs**

- **Latent MoE** — routes continuous hidden states (not raw tokens) to specialist experts; 550B of knowledge, 55B activated per pass.
- **Multi-Token Prediction (MTP)** — predicts several future tokens per forward pass, lifting tokens-per-second and improving reasoning planning.
- **NVFP4 pretraining** — 4-bit floating-point training doubles throughput vs 8-bit, enabling a richer training set.

Endpoint: `{NVIDIA_BASE_URL}`  ·  Model: `{MODEL}`  ·  OpenAI-compatible API.
"""
            )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, server_name="0.0.0.0", server_port=7860)
