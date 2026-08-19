"""Two numeric inputs feeding a single output."""

import gradio as gr

TITLE = "Unit Converter"
DESCRIPTION = "Multiple inputs, including a dropdown."
ORDER = 20

FACTORS = {
    "millimetres -> inches": 1 / 25.4,
    "inches -> millimetres": 25.4,
    "kilograms -> pounds": 2.20462,
    "pounds -> kilograms": 1 / 2.20462,
}


def convert(value, conversion):
    return round(value * FACTORS[conversion], 4)


demo = gr.Interface(
    fn=convert,
    inputs=[
        gr.Number(value=100, label="Value"),
        gr.Dropdown(list(FACTORS), value="millimetres -> inches", label="Conversion"),
    ],
    outputs=gr.Number(label="Result"),
    title=TITLE,
)
