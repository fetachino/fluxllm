"""FluxLLM demo page: single-request streaming + concurrent burst view.

Served by the API app at ``/demo`` when the ``demo/`` directory ships with
the deployment (FastAPI StaticFiles mount in ``api.app``). The page talks
only to public HTTP endpoints:

- ``POST /v1/chat/completions`` (stream=true, OpenAI-style SSE) for
  generation with a visible TTFT and token rate;
- ``GET /stats`` for live scheduler/KV state;
- ``GET /metrics`` for Prometheus gauges (batch size, utilization,
  prefix-cache hit rate);
- aborting the fetch cancels the request server-side (client disconnect
  handling in the API layer).
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def mount_demo(app: FastAPI) -> bool:
    """Serve the demo page if it is present next to the package.

    Returns True when mounted. Skipped silently in images that do not
    ship the demo directory (benchmark image, slim installs).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # repo root when run from a checkout, /app inside the container image
    candidates = [
        os.path.join(here, "..", "..", "demo"),
        os.path.join(here, "..", "demo"),
    ]
    for candidate in candidates:
        directory = os.path.abspath(candidate)
        if os.path.isdir(directory):
            app.mount(
                "/demo",
                StaticFiles(directory=directory, html=True),
                name="demo",
            )
            return True
    return False
