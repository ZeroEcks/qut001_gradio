import ast
import html
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

from qut001.constants import MODEL_WEIGHTS_DIR

MODEL_NAME = "Qwen/Qwen3-0.6B"
tokenizer = None
model = None


def load_model():
    global tokenizer, model
    model_path = os.path.join(MODEL_WEIGHTS_DIR, "qwen3_06b")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()


def get_predictions(context, top_k):
    inputs = tokenizer(context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)

    tokens = []
    probabilities = []
    token_ids = []
    for i in range(top_k):
        token = tokenizer.decode([top_indices[i]])
        tokens.append(token)
        probabilities.append(top_probs[i].item())
        token_ids.append(top_indices[i].item())

    return tokens, probabilities, token_ids


def build_bar_chart(tokens, probabilities):
    fig, ax = plt.subplots(figsize=(5, len(tokens) * 0.55 + 0.5))
    n = len(tokens)
    colors = ["#2196F3" if i == 0 else "#90CAF9" for i in range(n)]
    y_positions = range(n)
    ax.barh(y_positions, probabilities, color=colors, height=0.6)

    for bar, prob in zip(ax.containers[0], probabilities):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}",
            va="center",
            fontsize=11,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([repr(t) for t in tokens])
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_xlim(0, max(probabilities) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def format_generated_html(prompt, context):
    generated = context[len(prompt):]
    return (
        '<div style="font-family: monospace; font-size: 14px; line-height: 1.6; '
        'padding: 8px; white-space: pre-wrap; word-wrap: break-word;">'
        f'<span style="color: #999;">{html.escape(prompt)}</span>{html.escape(generated)}</div>'
    )


def _as_rows(data):
    if data is None:
        return []
    if hasattr(data, "values"):
        return data.values.tolist()
    return data


def build_step_log_df(step_log):
    if not step_log:
        return []
    rows = []
    for entry in step_log:
        ctx = entry["context_before"]
        if len(ctx) > 80:
            ctx = "..." + ctx[-77:]
        rows.append([entry["step"], repr(entry["token"]), entry["prob"], ctx])
    return rows


def on_start(prompt, top_k, max_steps):
    if not prompt.strip():
        return (
            None, [], "Please enter a prompt.",
            gr.update(visible=False), gr.update(visible=False),
            gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True),
            gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),
            "", "", 0, [], top_k, max_steps, [],
        )

    tokens, probs, token_ids = get_predictions(prompt, top_k)
    fig = build_bar_chart(tokens, probs)
    df_data = [[i + 1, repr(t), f"{p:.2%}"] for i, (t, p) in enumerate(zip(tokens, probs))]
    generated_html = format_generated_html(prompt, prompt)

    return (
        fig, df_data, generated_html,
        gr.update(visible=False),
        gr.update(value=f"Step 1 / {max_steps}", visible=True),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        prompt, prompt, 1, [], top_k, max_steps, token_ids,
    )


def on_select(evt: gr.SelectData, df_data, original_prompt, context, step, step_log,
              top_k, max_steps, token_ids):
    if df_data is None or not original_prompt:
        return (
            gr.update(), df_data, gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            original_prompt, context, step, step_log, top_k, max_steps, token_ids,
        )

    df_data = _as_rows(df_data)
    if not df_data:
        return (
            gr.update(), df_data, gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            original_prompt, context, step, step_log, top_k, max_steps, token_ids,
        )

    row_idx = evt.index[0]
    token_repr = df_data[row_idx][1]
    prob_str = df_data[row_idx][2]
    token_text = ast.literal_eval(token_repr)

    new_context = context + token_text
    new_step = step + 1
    new_log = step_log + [{
        "step": step,
        "token": token_text,
        "prob": prob_str,
        "context_before": context,
    }]

    if new_step > max_steps:
        generated_html = format_generated_html(original_prompt, new_context)
        log_df = build_step_log_df(new_log)
        return (
            None, [],
            generated_html,
            gr.update(value=log_df, visible=True),
            gr.update(value=f"Done ({max_steps} steps)", visible=True),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            original_prompt, new_context, new_step, new_log, top_k, max_steps, [],
        )

    tokens, probs, new_token_ids = get_predictions(new_context, top_k)
    fig = build_bar_chart(tokens, probs)
    new_df_data = [[i + 1, repr(t), f"{p:.2%}"] for i, (t, p) in enumerate(zip(tokens, probs))]
    generated_html = format_generated_html(original_prompt, new_context)
    log_df = build_step_log_df(new_log)

    return (
        fig, new_df_data,
        generated_html,
        gr.update(value=log_df, visible=True),
        gr.update(value=f"Step {new_step} / {max_steps}"),
        gr.update(interactive=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        original_prompt, new_context, new_step, new_log, top_k, max_steps, new_token_ids,
    )


def on_stop():
    return (
        gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True),
        gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),
    )


def on_reset():
    return (
        None, [], "",
        gr.update(visible=False), gr.update(visible=False),
        gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True),
        gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True),
        "", "", 0, [], 5, 40, [],
    )


with gr.Blocks(title="LLM Next-Token Prediction Demo") as demo:
    gr.Markdown("## LLM Next-Token Prediction Demo")
    gr.Markdown(
        "Enter a prompt, then click on a token in the list to append it. "
        "Watch the model's top predictions update autoregressively at each step."
    )

    with gr.Row():
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="Enter your prompt here...",
            scale=3,
        )
        top_k_input = gr.Dropdown(
            choices=list(range(5, 21)),
            value=5,
            label="Top-K",
            scale=1,
        )
        max_steps_input = gr.Slider(
            minimum=1,
            maximum=100,
            value=40,
            step=1,
            label="Max Steps",
            scale=1,
        )

    with gr.Row():
        start_btn = gr.Button("Start", variant="primary", scale=1)
        stop_btn = gr.Button("Stop", variant="stop", interactive=False, scale=1)
        reset_btn = gr.Button("Reset", variant="secondary", interactive=True, scale=1)

    step_label = gr.Markdown(visible=False)

    with gr.Row():
        bar_chart = gr.Plot(label="Top-K Probabilities", scale=1)
        token_table = gr.Dataframe(
            headers=["Rank", "Token", "Probability"],
            label="Click a row to select the next token",
            interactive=False,
            scale=1,
        )

    generated_text = gr.HTML(label="Generated Text")

    step_log_table = gr.Dataframe(
        headers=["Step", "Token", "Probability", "Context Before"],
        label="Generation Log",
        visible=False,
    )

    original_prompt_state = gr.State("")
    context_state = gr.State("")
    step_state = gr.State(0)
    step_log_state = gr.State([])
    top_k_state = gr.State(5)
    max_steps_state = gr.State(40)
    token_ids_state = gr.State([])

    demo.load(fn=load_model, outputs=None)

    start_btn.click(
        fn=on_start,
        inputs=[prompt_input, top_k_input, max_steps_input],
        outputs=[
            bar_chart, token_table, generated_text,
            step_log_table, step_label,
            start_btn, stop_btn, reset_btn,
            prompt_input, top_k_input, max_steps_input,
            original_prompt_state, context_state, step_state,
            step_log_state, top_k_state, max_steps_state,
            token_ids_state,
        ],
    )

    token_table.select(
        fn=on_select,
        inputs=[token_table,
                original_prompt_state, context_state, step_state,
                step_log_state, top_k_state, max_steps_state, token_ids_state],
        outputs=[
            bar_chart, token_table, generated_text,
            step_log_table, step_label,
            start_btn, stop_btn, reset_btn,
            prompt_input, top_k_input, max_steps_input,
            original_prompt_state, context_state, step_state,
            step_log_state, top_k_state, max_steps_state,
            token_ids_state,
        ],
    )

    stop_btn.click(
        fn=on_stop,
        inputs=None,
        outputs=[start_btn, stop_btn, reset_btn, prompt_input, top_k_input, max_steps_input],
    )

    reset_btn.click(
        fn=on_reset,
        inputs=None,
        outputs=[
            bar_chart, token_table, generated_text,
            step_log_table, step_label,
            start_btn, stop_btn, reset_btn,
            prompt_input, top_k_input, max_steps_input,
            original_prompt_state, context_state, step_state,
            step_log_state, top_k_state, max_steps_state,
            token_ids_state,
        ],
    )


if __name__ == "__main__":
    demo.launch(share=True,
                server_name="0.0.0.0",
                server_port=7860)
