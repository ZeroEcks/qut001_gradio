"""The same idea built with Blocks instead of Interface.

Blocks gives you control over layout, which is what you want once a demo grows
beyond a single row of inputs.
"""

import gradio as gr

TITLE = "Text Tools"
DESCRIPTION = "Built with Blocks for custom layout."
ORDER = 30


def analyse(text):
    return text[::-1], f"{len(text)} characters, {len(text.split())} words"


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")

    with gr.Row():
        source = gr.Textbox(label="Text", lines=4)
        with gr.Column():
            reversed_text = gr.Textbox(label="Reversed", lines=4)
            stats = gr.Textbox(label="Statistics")

    gr.Button("Analyse", variant="primary").click(
        analyse, inputs=source, outputs=[reversed_text, stats]
    )
