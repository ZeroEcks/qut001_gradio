"""Simplest possible app: one input, one output.

The module-level names below are what the hub reads to build the home page.
"""

import gradio as gr

TITLE = "Greeter"
DESCRIPTION = "A one-input, one-output Interface."
ORDER = 10


def greet(name):
    return f"Hello, {name}!"


demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(label="Your name"),
    outputs=gr.Textbox(label="Greeting"),
    title=TITLE,
    examples=["Ada", "Grace"],
)
