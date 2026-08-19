"""App discovery.

Finds every Gradio app in the ``apps/`` package so that adding a new demo
requires nothing more than dropping a new ``.py`` file into that folder.

A module is treated as an app if it defines a module-level ``demo`` object
(a ``gr.Blocks`` or ``gr.Interface``). These optional module-level variables
customise how it appears in the hub:

    TITLE       -- display name        (default: derived from the filename)
    DESCRIPTION -- one-line summary    (default: "")
    SLUG        -- URL path segment    (default: filename with _ replaced by -)
    ORDER       -- sort position       (default: 100, lower appears first)

Modules whose names begin with ``_`` are skipped, which is how ``_template.py``
stays out of the running site.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
import traceback
from dataclasses import dataclass, field

import gradio as gr

import apps

logger = logging.getLogger(__name__)


@dataclass
class LoadedApp:
    """An app module that imported successfully."""

    slug: str
    title: str
    description: str
    order: int
    demo: gr.Blocks


@dataclass
class BrokenApp:
    """An app module that raised on import.

    Kept rather than discarded so the hub can display the traceback instead of
    silently hiding the demo.
    """

    module: str
    error: str
    traceback: str = field(repr=False, default="")


def _humanise(module_name: str) -> str:
    return module_name.replace("_", " ").title()


def discover_apps() -> tuple[list[LoadedApp], list[BrokenApp]]:
    """Import every app module and return the ones that worked and the ones that didn't.

    A failing module is reported rather than raised, so one broken demo never
    prevents the rest of the hub from starting.
    """
    loaded: list[LoadedApp] = []
    broken: list[BrokenApp] = []

    for module_info in sorted(pkgutil.iter_modules(apps.__path__), key=lambda m: m.name):
        name = module_info.name
        if name.startswith("_"):
            continue

        try:
            module = importlib.import_module(f"apps.{name}")
        except Exception as exc:
            logger.error("Failed to import apps.%s: %s", name, exc)
            broken.append(
                BrokenApp(module=name, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
            )
            continue

        demo = getattr(module, "demo", None)
        if demo is None:
            logger.warning("Skipping apps.%s: no module-level 'demo' object found.", name)
            broken.append(
                BrokenApp(module=name, error="No module-level 'demo' object was defined.")
            )
            continue

        loaded.append(
            LoadedApp(
                slug=getattr(module, "SLUG", name.replace("_", "-")),
                title=getattr(module, "TITLE", _humanise(name)),
                description=getattr(module, "DESCRIPTION", ""),
                order=getattr(module, "ORDER", 100),
                demo=demo,
            )
        )

    loaded.sort(key=lambda a: (a.order, a.title))
    return loaded, broken
