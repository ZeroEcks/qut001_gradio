# Engineering Gradio Demos

A template for hosting several Gradio demos from a single container. Each demo
lives in its own file under `apps/` and is served at its own URL, with a
generated home page linking to all of them.

## Quick start

```bash
docker compose up --build
```

Then open <http://localhost:7860>.

Because `docker-compose.yml` mounts the source into the container and runs
uvicorn with `--reload`, editing a file in `apps/` refreshes the site a second
or two later. No rebuild is needed unless you change `requirements.txt`.

## Adding a demo

```bash
cp apps/_template.py apps/my_demo.py
```

Edit the new file and it appears on the home page automatically. There is no
registry or list to update.

A file in `apps/` becomes a demo when it defines a module-level `demo`:

```python
import gradio as gr

TITLE = "My Demo"
DESCRIPTION = "One line for the home page card."
ORDER = 40

demo = gr.Interface(fn=lambda x: x * 2, inputs="number", outputs="number")
```

| Variable | Required | Purpose |
| --- | --- | --- |
| `demo` | yes | The `gr.Interface` or `gr.Blocks` object to serve. |
| `TITLE` | no | Card heading. Defaults to the filename. |
| `DESCRIPTION` | no | Card subtitle. Defaults to empty. |
| `ORDER` | no | Sort position, lower first. Defaults to `100`. |
| `SLUG` | no | URL segment. Defaults to the filename with `_` as `-`. |

Files starting with `_` are ignored, which is why `_template.py` does not show
up on the site.

## When a demo has an error

A demo that fails to import does not stop the others. The home page lists the
broken file with its traceback, and the same error appears in the logs:

```bash
docker compose logs -f
```

This is deliberate. Mid-edit syntax errors are common, and they should not take
the whole site down during a lecture.

## Published images

Pushing to `main` builds the container and publishes it to the GitHub Container
Registry, tagged with the Unix timestamp of the build:

```bash
docker pull ghcr.io/<owner>/<repo>:latest        # newest build
docker pull ghcr.io/<owner>/<repo>:1785718372    # one specific build
docker run -p 7860:7860 ghcr.io/<owner>/<repo>:latest
```

Timestamp tags are never overwritten, so pinning one guarantees a class gets the
same demos all semester regardless of later changes. Since the number only
increases, sorting tags numerically sorts them by build order. The exact `pull`
command for each build appears in the workflow run summary.

No secrets are needed; the workflow uses the automatic `GITHUB_TOKEN`. The
package starts out private, so to let students pull it without signing in, go to
**Repository → Packages → Package settings → Change visibility → Public**.

Pull requests build the image and run the checks but do not publish, so a broken
Dockerfile is caught before it reaches `main`.

## Running without Docker

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-torch.txt
pip install -r requirements.txt
uvicorn main:app --reload
```

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `SITE_TITLE` | `Engineering Gradio Demos` | Heading on the home page. |
| `LOG_LEVEL` | `INFO` | Python logging level. |

## Dependencies

Versions in `requirements.txt` are pinned on purpose. Gradio changes its API
between major versions, so an unpinned install can break working demos when the
image is rebuilt later. To upgrade, bump the versions, rebuild, and check each
demo still runs.

## Layout

```
.
├── .github/workflows/
│   └── docker-publish.yml  builds and pushes the image to GHCR
├── apps/
│   ├── _template.py        copy this to start a new demo
│   ├── greeter.py          Interface, single input
│   ├── unit_converter.py   Interface, multiple inputs
│   └── text_tools.py       Blocks, custom layout
├── hub.py                  finds the demos in apps/
├── main.py                 home page and mounting
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
└── requirements-torch.txt
```

Most of the time only `apps/` needs to change.
