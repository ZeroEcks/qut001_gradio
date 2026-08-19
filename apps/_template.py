"""Starting point for a new demo.

To use it:

    cp apps/_template.py apps/my_demo.py

then edit the file. The leading underscore is what keeps THIS file hidden from
the site; once you rename it without one, it is picked up automatically the
next time the server reloads. There is nothing else to register.

Only ``demo`` is required. TITLE, DESCRIPTION and ORDER just control how the
card looks on the home page.
"""

import gradio as gr

TITLE = "My Demo"
DESCRIPTION = "One line describing what students should notice."
ORDER = 100  # Lower numbers appear first on the home page.


def run(value):
    return value * 2


demo = gr.Interface(
    fn=run,
    inputs=gr.Number(value=21, label="Input"),
    outputs=gr.Number(label="Output"),
    title=TITLE,
)
