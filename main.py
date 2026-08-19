"""FastAPI host that mounts every discovered Gradio app under a single server.

Run locally with:      uvicorn main:app --reload
Run in Docker with:    docker compose up
"""

from __future__ import annotations

import html
import logging
import os

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from hub import BrokenApp, LoadedApp, discover_apps

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("hub.main")

SITE_TITLE = os.getenv("SITE_TITLE", "Engineering Gradio Demos")

app = FastAPI(title=SITE_TITLE)

loaded_apps, broken_apps = discover_apps()


def _render_card(item: LoadedApp) -> str:
    description = (
        f'<p class="desc">{html.escape(item.description)}</p>' if item.description else ""
    )
    return f"""
      <a class="card" href="/{html.escape(item.slug)}/">
        <h2>{html.escape(item.title)}</h2>
        {description}
      </a>"""


def _render_broken(item: BrokenApp) -> str:
    detail = (
        f"<pre>{html.escape(item.traceback)}</pre>" if item.traceback else ""
    )
    return f"""
      <details class="broken">
        <summary><strong>apps/{html.escape(item.module)}.py</strong> &mdash; {html.escape(item.error)}</summary>
        {detail}
      </details>"""


def _render_index() -> str:
    if loaded_apps:
        cards = "".join(_render_card(item) for item in loaded_apps)
    else:
        cards = (
            '<p class="empty">No apps found yet. Copy <code>apps/_template.py</code> '
            "to <code>apps/my_demo.py</code> to add the first one.</p>"
        )

    problems = ""
    if broken_apps:
        problems = (
            '<section class="problems"><h3>These apps failed to load</h3>'
            + "".join(_render_broken(item) for item in broken_apps)
            + "</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{html.escape(SITE_TITLE)}</title>
    <style>
      :root {{ color-scheme: light dark; }}
      body {{
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
        margin: 0 auto; padding: 3rem 1.5rem; max-width: 60rem; line-height: 1.5;
      }}
      h1 {{ margin-bottom: .25rem; }}
      .subtitle {{ margin-top: 0; opacity: .7; }}
      .grid {{
        display: grid; gap: 1rem; margin-top: 2rem;
        grid-template-columns: repeat(auto-fill, minmax(16rem, 1fr));
      }}
      .card {{
        display: block; padding: 1.25rem; border: 1px solid rgba(128,128,128,.35);
        border-radius: .75rem; text-decoration: none; color: inherit;
        transition: border-color .15s ease, transform .15s ease;
      }}
      .card:hover {{ border-color: #2563eb; transform: translateY(-2px); }}
      .card h2 {{ margin: 0 0 .4rem; font-size: 1.1rem; }}
      .desc {{ margin: 0; font-size: .9rem; opacity: .75; }}
      .problems {{ margin-top: 3rem; }}
      .broken {{
        border-left: 3px solid #dc2626; padding: .6rem .9rem;
        margin-bottom: .6rem; background: rgba(220,38,38,.07);
      }}
      pre {{ overflow-x: auto; font-size: .8rem; }}
    </style>
  </head>
  <body>
    <h1>{html.escape(SITE_TITLE)}</h1>
    <p class="subtitle">Interactive demonstrations for engineering coursework.</p>
    <div class="grid">{cards}
    </div>
    {problems}
  </body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _render_index()


@app.get("/healthz")
def healthz() -> dict:
    """Liveness probe used by the Docker HEALTHCHECK."""
    return {
        "status": "ok",
        "apps_loaded": len(loaded_apps),
        "apps_failed": len(broken_apps),
    }


for item in loaded_apps:
    app = gr.mount_gradio_app(app, item.demo, path=f"/{item.slug}")
    logger.info("Mounted %-28s -> /%s", item.title, item.slug)

if broken_apps:
    for item in broken_apps:
        logger.error("Could not load apps/%s.py: %s", item.module, item.error)
