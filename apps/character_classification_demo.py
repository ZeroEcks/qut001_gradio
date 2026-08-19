"""Interactive Gradio app for teaching AI inputs, outputs, and confidence."""

from __future__ import annotations

import argparse
import html
from functools import lru_cache
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps

from qut001.character_classification_model import (
    CLASS_NAMES,
    DIGIT_NAMES,
    INPUT_MEAN,
    INPUT_STD,
    LETTER_NAMES,
    load_model,
)

CANVAS_SIZE = 420
INTENTION_OPTIONS = [
    "Not specified",
    *CLASS_NAMES,
    "Ambiguous between characters",
    "Not a single digit or letter",
]

CHALLENGES = {
    "1. Clear character": (
        "Draw one clear digit or letter. Predict the model's answer before you "
        "press **Make prediction**. Lowercase and uppercase letters share a class."
    ),
    "2. Look-alike characters": (
        "Try a confusing pair such as 0/O, 1/I/L, 2/Z, 5/S, 8/B, or G/Q. "
        "Inspect the top competing output scores."
    ),
    "3. Change one thing": (
        "Change only one feature—colour, thickness, size, position, or letter "
        "case—and compare the new output with the previous prediction."
    ),
    "4. Outside the task": (
        "Draw punctuation, an emoji, a shape, several characters, or a scribble. "
        "Choose **Not a single digit or letter**. The model must still select "
        "one of its 36 available classes."
    ),
}

APP_CSS = """
:root {
    --qut-blue: #005ea8;
    --deep-blue: #093b66;
    --soft-blue: #edf6fc;
    --soft-grey: #f5f7f9;
    --ink: #17212b;
}
.gradio-container { max-width: 1550px !important; }
.hero {
    background: linear-gradient(120deg, var(--deep-blue), var(--qut-blue));
    color: white;
    padding: 1.4rem 1.6rem;
    border-radius: 16px;
    margin-bottom: 1rem;
}
.hero h1 { margin: 0 0 .35rem 0; font-size: 2rem; }
.hero p { margin: 0; font-size: 1.05rem; max-width: 1100px; }
.stage-label { font-weight: 750; font-size: 1.05rem; margin-bottom: .4rem; }
.arrow-card {
    display: flex; justify-content: center; align-items: center;
    min-height: 220px; font-size: 2.6rem; color: var(--qut-blue); font-weight: 800;
}
.model-card {
    border: 2px solid var(--qut-blue); background: var(--soft-blue);
    border-radius: 14px; padding: 1rem; min-height: 210px;
    display: flex; flex-direction: column; justify-content: center; text-align: center;
}
.model-card .model-icon { font-size: 2.5rem; }
.model-card strong { font-size: 1.15rem; }
.output-card {
    border: 1px solid #d7dee5; border-radius: 14px;
    padding: .9rem 1rem; background: white;
}
.prediction-headline { font-size: 1.2rem; font-weight: 750; margin-bottom: .35rem; }
.group-summary { font-size: .9rem; color: #44515e; margin-bottom: .7rem; }
.pred-row {
    display: grid; grid-template-columns: 2.1rem 1fr 4rem;
    align-items: center; gap: .55rem; margin: .28rem 0;
}
.pred-track { background: #e9edf1; border-radius: 999px; overflow: hidden; height: 1rem; }
.pred-fill { background: #7b8794; height: 100%; min-width: 0; transition: width .25s ease; }
.pred-row.top .pred-fill { background: var(--qut-blue); }
.pred-row.top .character, .pred-row.top .pct { font-weight: 800; }
.notice { border-left: 5px solid var(--qut-blue); background: var(--soft-blue); padding: .85rem 1rem; border-radius: 8px; }
.warning { border-left-color: #ad5b00; background: #fff5e8; }
.success { border-left-color: #19733b; background: #edf9f1; }
.comparison { background: var(--soft-grey); padding: .8rem 1rem; border-radius: 10px; }
.pixel-note { font-size: .93rem; background: var(--soft-grey); padding: .65rem .8rem; border-radius: 9px; }
.challenge-box { background: #f8fbfe; border: 1px solid #cfe3f2; padding: .85rem 1rem; border-radius: 10px; }
.score-section-title { font-weight: 800; margin: .7rem 0 .35rem; }
.score-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(88px, 1fr)); gap: .4rem; }
.score-chip { border: 1px solid #d7dee5; border-radius: 8px; padding: .42rem .5rem; background: white; display:flex; justify-content:space-between; gap:.35rem; }
.score-chip.top { border: 2px solid var(--qut-blue); background: var(--soft-blue); font-weight: 800; }
@media (max-width: 900px) {
    .arrow-card { min-height: auto; transform: rotate(90deg); padding: .3rem; }
}
"""


def blank_canvas() -> Image.Image:
    return Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), color="black")


def blank_model_preview() -> Image.Image:
    return Image.new("L", (280, 280), color=0)


def editor_to_rgb(editor_value: Any) -> Image.Image:
    """Extract the composite image from a Gradio ImageEditor value."""

    if editor_value is None:
        return blank_canvas()

    value = editor_value
    if isinstance(editor_value, dict):
        value = editor_value.get("composite")
        if value is None:
            value = editor_value.get("background")

    if value is None:
        return blank_canvas()
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        array = value
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")

    raise TypeError(f"Unsupported editor value type: {type(value)!r}")


def shift_with_zeros(image: np.ndarray, shift_y: int, shift_x: int) -> np.ndarray:
    """Translate an image without wrapping pixels around the edges."""

    output = np.zeros_like(image)
    source_y0 = max(0, -shift_y)
    source_y1 = min(image.shape[0], image.shape[0] - shift_y)
    source_x0 = max(0, -shift_x)
    source_x1 = min(image.shape[1], image.shape[1] - shift_x)
    target_y0 = max(0, shift_y)
    target_x0 = max(0, shift_x)
    target_y1 = target_y0 + max(0, source_y1 - source_y0)
    target_x1 = target_x0 + max(0, source_x1 - source_x0)

    if source_y1 > source_y0 and source_x1 > source_x0:
        output[target_y0:target_y1, target_x0:target_x1] = image[
            source_y0:source_y1, source_x0:source_x1
        ]
    return output


def centre_and_resize(grayscale: Image.Image) -> np.ndarray:
    """Convert a drawing to an EMNIST-like centred 28 × 28 image."""

    source = np.asarray(grayscale, dtype=np.uint8)
    active = source > 8
    if not np.any(active):
        return np.zeros((28, 28), dtype=np.uint8)

    ys, xs = np.where(active)
    crop = grayscale.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    width, height = crop.size
    scale = min(20 / max(width, 1), 20 / max(height, 1))
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    crop = crop.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - resized_width) // 2
    top = (28 - resized_height) // 2
    canvas.paste(crop, (left, top))

    array = np.asarray(canvas, dtype=np.uint8)
    weights = array.astype(np.float32)
    total = float(weights.sum())
    if total > 0:
        y_grid, x_grid = np.indices(array.shape)
        centre_y = float((y_grid * weights).sum() / total)
        centre_x = float((x_grid * weights).sum() / total)
        array = shift_with_zeros(
            array,
            shift_y=int(round(13.5 - centre_y)),
            shift_x=int(round(13.5 - centre_x)),
        )
    return array


def preprocess(editor_value: Any, mode: str) -> tuple[np.ndarray, Image.Image, str]:
    """Create the model input, enlarged preview, and input-description HTML."""

    rgb = editor_to_rgb(editor_value)
    grayscale = ImageOps.grayscale(rgb)

    if mode == "Resize the whole canvas":
        model_array = np.asarray(
            grayscale.resize((28, 28), Image.Resampling.LANCZOS), dtype=np.uint8
        )
    else:
        model_array = centre_and_resize(grayscale)

    preview = Image.fromarray(model_array).resize((280, 280), Image.Resampling.NEAREST)
    lit_pixels = int((model_array > 16).sum())
    max_intensity = int(model_array.max())
    input_note = (
        '<div class="pixel-note"><strong>The model receives:</strong> '
        '28 × 28 grayscale pixels = <strong>784 numbers</strong>.<br>'
        f'Pixels with visible signal: <strong>{lit_pixels}</strong>/784; '
        f'brightest value: <strong>{max_intensity}</strong>/255. '
        "Drawing colour has been converted to brightness.</div>"
    )
    return model_array, preview, input_note


@lru_cache(maxsize=1)
def get_model() -> torch.nn.Module:
    return load_model("cpu")


def empty_prediction_html(message: str = "No prediction yet") -> str:
    rows = "".join(
        (
            '<div class="pred-row">'
            f'<span class="character">{name}</span>'
            '<div class="pred-track"><div class="pred-fill" style="width:0%"></div></div>'
            '<span class="pct">—</span></div>'
        )
        for name in CLASS_NAMES[:8]
    )
    return (
        '<div class="output-card">'
        f'<div class="prediction-headline">{html.escape(message)}</div>'
        '<div class="group-summary">The highest-ranked outputs will appear here.</div>'
        f"{rows}</div>"
    )


def empty_full_scores_html() -> str:
    return '<div class="pixel-note">All 36 class scores will appear after prediction.</div>'


def prediction_html(probabilities: np.ndarray) -> str:
    ranking = np.argsort(probabilities)[::-1]
    top_index = int(ranking[0])
    top_name = CLASS_NAMES[top_index]
    top_score = float(probabilities[top_index])
    digit_total = float(probabilities[:10].sum())
    letter_total = float(probabilities[10:].sum())

    rows: list[str] = []
    for rank, class_index in enumerate(ranking[:10]):
        name = CLASS_NAMES[int(class_index)]
        score = float(probabilities[class_index])
        top_class = " top" if rank == 0 else ""
        width = max(0.0, min(100.0, score * 100.0))
        rows.append(
            f'<div class="pred-row{top_class}">'
            f'<span class="character">{name}</span>'
            '<div class="pred-track">'
            f'<div class="pred-fill" style="width:{width:.2f}%"></div>'
            "</div>"
            f'<span class="pct">{score:.1%}</span></div>'
        )

    return (
        '<div class="output-card">'
        f'<div class="prediction-headline">Top prediction: {top_name} '
        f'<span style="font-weight:500">({top_score:.1%} model score)</span></div>'
        f'<div class="group-summary">Total score assigned to digits: {digit_total:.1%} · '
        f'letters: {letter_total:.1%}<br>Showing the ten highest of 36 outputs.</div>'
        + "".join(rows)
        + "</div>"
    )


def full_scores_html(probabilities: np.ndarray) -> str:
    top_index = int(np.argmax(probabilities))

    def chips(indices: range) -> str:
        parts = []
        for index in indices:
            top_class = " top" if index == top_index else ""
            parts.append(
                f'<div class="score-chip{top_class}"><strong>{CLASS_NAMES[index]}</strong>'
                f'<span>{float(probabilities[index]):.1%}</span></div>'
            )
        return "".join(parts)

    return (
        '<div class="score-section-title">Digits</div>'
        f'<div class="score-grid">{chips(range(0, 10))}</div>'
        '<div class="score-section-title">Letters (case-insensitive)</div>'
        f'<div class="score-grid">{chips(range(10, 36))}</div>'
    )


def interpretation_html(probabilities: np.ndarray, intention: str) -> str:
    ranking = np.argsort(probabilities)[::-1]
    top_index = int(ranking[0])
    second_index = int(ranking[1])
    top_name = CLASS_NAMES[top_index]
    second_name = CLASS_NAMES[second_index]
    top_score = float(probabilities[top_index])
    second_score = float(probabilities[second_index])
    margin = top_score - second_score

    if intention == "Not a single digit or letter":
        outcome = (
            '<div class="notice warning"><strong>You said this was outside the task.</strong> '
            f'The model still selected <strong>{top_name}</strong> because its only allowed '
            "outputs are 0–9 and A–Z. A strong score does not prove that the input belongs "
            "to the task.</div>"
        )
    elif intention == "Ambiguous between characters":
        outcome = (
            '<div class="notice"><strong>You intended an ambiguous character.</strong> '
            f'The model ranked <strong>{top_name}</strong> first and <strong>{second_name}</strong> '
            "second. Compare their scores and consider whether the output reflects the ambiguity.</div>"
        )
    elif intention in CLASS_NAMES:
        intended_index = CLASS_NAMES.index(intention)
        if top_index == intended_index:
            outcome = (
                '<div class="notice success"><strong>The output matched your intention.</strong> '
                f'You intended {intention}, and the model ranked it first. Uppercase and '
                "lowercase forms are treated as the same letter class.</div>"
            )
        else:
            intended_score = float(probabilities[intended_index])
            outcome = (
                '<div class="notice warning"><strong>The output did not match your intention.</strong> '
                f'You intended {intention}, but the model selected {top_name}. It assigned '
                f'{intended_score:.1%} to your intended class.</div>'
            )
    else:
        outcome = (
            '<div class="notice"><strong>What does the score mean?</strong> '
            f'The model favoured {top_name} over {second_name} by {margin:.1%}. '
            "This describes its relative output distribution, not a guarantee of correctness.</div>"
        )

    detail = (
        '<div class="pixel-note" style="margin-top:.6rem">'
        f'Second choice: <strong>{second_name}</strong> ({second_score:.1%}). '
        f'Top-two gap: <strong>{margin:.1%}</strong>. '
        "A small gap often indicates competing interpretations.</div>"
    )
    return outcome + detail


def comparison_html(previous: list[float] | None, current: np.ndarray) -> str:
    if previous is None:
        return (
            '<div class="comparison"><strong>First prediction recorded.</strong> '
            "Change one part of the input, then predict again to compare the outputs.</div>"
        )

    previous_array = np.asarray(previous, dtype=np.float32)
    previous_top = int(np.argmax(previous_array))
    current_top = int(np.argmax(current))
    previous_score = float(previous_array[previous_top])
    current_score = float(current[current_top])
    previous_name = CLASS_NAMES[previous_top]
    current_name = CLASS_NAMES[current_top]

    if previous_top == current_top:
        delta = current_score - previous_score
        direction = "increased" if delta >= 0 else "decreased"
        return (
            '<div class="comparison"><strong>The top prediction stayed the same:</strong> '
            f'{current_name}. Its score {direction} from {previous_score:.1%} to '
            f'{current_score:.1%} ({delta:+.1%}).</div>'
        )

    return (
        '<div class="comparison"><strong>The model changed its mind:</strong> '
        f'{previous_name} ({previous_score:.1%}) → {current_name} ({current_score:.1%}). '
        "Which input change may have caused this?</div>"
    )


def predict(
        editor_value: Any,
        preprocessing_mode: str,
        intention: str,
        previous_probabilities: list[float] | None,
) -> tuple[Image.Image, str, str, str, str, str, list[float] | None]:
    try:
        model_array, preview, input_note = preprocess(editor_value, preprocessing_mode)
    except Exception as exc:
        return (
            blank_model_preview(),
            '<div class="notice warning">Could not read the drawing.</div>',
            empty_prediction_html("Prediction unavailable"),
            empty_full_scores_html(),
            f'<div class="notice warning">{html.escape(str(exc))}</div>',
            "",
            previous_probabilities,
        )

    if int(model_array.max()) < 12 or int(model_array.sum()) < 100:
        return (
            preview,
            input_note,
            empty_prediction_html("The input appears blank"),
            empty_full_scores_html(),
            '<div class="notice warning"><strong>Draw something first.</strong> '
            "A blank image is still an input, but this activity works better after you add a mark.</div>",
            "",
            previous_probabilities,
        )

    tensor = torch.from_numpy(model_array.astype(np.float32) / 255.0)
    tensor = (tensor - INPUT_MEAN) / INPUT_STD
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    try:
        model = get_model()
        with torch.inference_mode():
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    except FileNotFoundError as exc:
        return (
            preview,
            input_note,
            empty_prediction_html("Model setup required"),
            empty_full_scores_html(),
            '<div class="notice warning"><strong>The interface is ready, but the trained '
            f'model is missing.</strong><br>{html.escape(str(exc))}</div>',
            "",
            previous_probabilities,
        )
    except Exception as exc:
        return (
            preview,
            input_note,
            empty_prediction_html("Prediction failed"),
            empty_full_scores_html(),
            f'<div class="notice warning">{html.escape(str(exc))}</div>',
            "",
            previous_probabilities,
        )

    return (
        preview,
        input_note,
        prediction_html(probabilities),
        full_scores_html(probabilities),
        interpretation_html(probabilities, intention),
        comparison_html(previous_probabilities, probabilities),
        probabilities.astype(float).tolist(),
    )


def reset_app() -> tuple[Image.Image, Image.Image, str, str, str, str, str, None]:
    return (
        blank_canvas(),
        blank_model_preview(),
        '<div class="pixel-note">The processed 28 × 28 input will appear here.</div>',
        empty_prediction_html(),
        empty_full_scores_html(),
        '<div class="notice">Draw one character, state what you intended, and make a prediction.</div>',
        "",
        None,
    )


def challenge_text(challenge: str) -> str:
    text = CHALLENGES.get(challenge, CHALLENGES["1. Clear character"])
    return f'<div class="challenge-box"><strong>Your task:</strong> {text}</div>'


# def build_demo() -> gr.Blocks:
with gr.Blocks(title="What Does an AI Model See?") as demo:
    gr.HTML(
        """
        <div class="hero">
          <h1>What does an AI model see?</h1>
          <p>Draw one digit or letter, inspect the exact 28 × 28 image received by
          the model, and observe how its 36 output scores change.</p>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=5, min_width=310):
            gr.HTML('<div class="stage-label">1. Create an input</div>')
            drawing = gr.ImageEditor(
                value=blank_canvas(),
                label="Draw here",
                show_label=False,
                type="pil",
                image_mode="RGB",
                sources=[],
                canvas_size=(CANVAS_SIZE, CANVAS_SIZE),
                fixed_canvas=True,
                transforms=(),
                layers=False,
                brush=gr.Brush(
                    default_size=24,
                    colors=[
                        "#FFFFFF",
                        "#FF5252",
                        "#FFD740",
                        "#69F0AE",
                        "#40C4FF",
                        "#7C4DFF",
                    ],
                    default_color="#FFFFFF",
                    color_mode="defaults",
                ),
                eraser=gr.Eraser(default_size=32),
                height=440,
            )
            intention = gr.Dropdown(
                choices=INTENTION_OPTIONS,
                value="Not specified",
                label="What did you intend to draw?",
                info=(
                    "Choose A for either uppercase or lowercase a. This answer is "
                    "used only for discussion; it is not given to the model."
                ),
            )
            preprocessing_mode = gr.Radio(
                choices=["Centre and resize", "Resize the whole canvas"],
                value="Centre and resize",
                label="Input preprocessing",
                info="Try both modes to see how preprocessing changes the model input.",
            )
            with gr.Row():
                predict_button = gr.Button("Make prediction", variant="primary", scale=3)
                clear_button = gr.Button("Clear", scale=1)

        with gr.Column(scale=1, min_width=60):
            gr.HTML('<div class="arrow-card">→</div>')

        with gr.Column(scale=3, min_width=260):
            gr.HTML('<div class="stage-label">2. Model input</div>')
            model_input = gr.Image(
                value=blank_model_preview(),
                label="Enlarged 28 × 28 grayscale input",
                show_label=True,
                interactive=False,
                height=300,
            )
            input_note = gr.HTML(
                '<div class="pixel-note">The processed 28 × 28 input will appear here.</div>'
            )

        with gr.Column(scale=1, min_width=60):
            gr.HTML('<div class="arrow-card">→</div>')

        with gr.Column(scale=3, min_width=250):
            gr.HTML('<div class="stage-label">3. AI model</div>')
            gr.HTML(
                """
                <div class="model-card">
                  <div class="model-icon">⚙️</div>
                  <strong>EMNIST character classifier</strong>
                  <span>A neural network trained beforehand on labelled handwritten
                  digits and letters.</span>
                  <hr style="width:80%; border:none; border-top:1px solid #b7d3e8">
                  <span>Available outputs: 0–9 and A–Z.<br>Letter case is merged.</span>
                  <span>It is not learning from your drawing.</span>
                </div>
                """
            )

        with gr.Column(scale=1, min_width=60):
            gr.HTML('<div class="arrow-card">→</div>')

        with gr.Column(scale=5, min_width=340):
            gr.HTML('<div class="stage-label">4. Output predictions</div>')
            prediction_output = gr.HTML(empty_prediction_html())

    with gr.Accordion("See all 36 output scores", open=False):
        full_prediction_output = gr.HTML(empty_full_scores_html())

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Interpret the result")
            interpretation = gr.HTML(
                '<div class="notice">Draw one character, state what you intended, and make a prediction.</div>'
            )
        with gr.Column(scale=1):
            gr.Markdown("### Compare with the previous input")
            comparison = gr.HTML("")

    gr.Markdown("## Guided experiments")
    with gr.Row():
        challenge = gr.Radio(
            choices=list(CHALLENGES),
            value="1. Clear character",
            label="Choose a challenge",
            scale=1,
        )
        challenge_prompt = gr.HTML(challenge_text("1. Clear character"), scale=2)

    with gr.Accordion("Discussion prompts", open=False):
        gr.Markdown(
            """
            - What was the model's actual input: the original canvas or the processed 28 × 28 image?
            - Did the highest model score always correspond to your intended character?
            - Which characters were easily confused, and why might their shapes overlap?
            - What changed when you altered colour, thickness, size, position, case, or preprocessing?
            - What happened when the input was not one single digit or letter?
            - In your discipline, who decides an AI system's inputs, available outputs, and acceptable errors?

            **Important:** the percentages are softmax output scores. They show how the model
            distributes preference among its 36 available classes; they are not guaranteed
            real-world probabilities and cannot establish that an input belongs to the task.
            """
        )

    previous_state = gr.State(None)

    predict_button.click(
        fn=predict,
        inputs=[drawing, preprocessing_mode, intention, previous_state],
        outputs=[
            model_input,
            input_note,
            prediction_output,
            full_prediction_output,
            interpretation,
            comparison,
            previous_state,
        ],
    )
    clear_button.click(
        fn=reset_app,
        inputs=[],
        outputs=[
            drawing,
            model_input,
            input_note,
            prediction_output,
            full_prediction_output,
            interpretation,
            comparison,
            previous_state,
        ],
    )
    challenge.change(fn=challenge_text, inputs=challenge, outputs=challenge_prompt)

# return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Use 0.0.0.0 to allow access from other devices on the same network.",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Ask Gradio to create a temporary public share link.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        css=APP_CSS)

    build_demo().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=APP_CSS,
    )
