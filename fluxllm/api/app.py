"""FastAPI application factory for FluxLLM.

The app is a thin shell: it builds the engine from settings at startup,
attaches the routes, and never touches internals beyond the public
FluxEngine API (submit/stream/stats/metrics).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from fluxllm.api.routes import router
from fluxllm.config.settings import EngineSettings
from fluxllm.metrics.metrics import global_metrics


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_app(settings: EngineSettings | None = None) -> FastAPI:
    settings = settings or EngineSettings()
    _configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    from fluxllm.api.demo import mount_demo

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from fluxllm.engine.engine import FluxEngine

        logger.info(
            "starting FluxLLM engine: model=%s deterministic=%s device=%s",
            settings.model_name if not settings.deterministic else "<deterministic>",
            settings.deterministic,
            settings.device,
        )
        engine = FluxEngine.from_settings(settings)
        app.state.engine = engine
        app.state.detokenize = engine._detokenize
        app.state.model_name = (
            settings.model_name if not settings.deterministic else "fluxllm-simulated"
        )
        app.state.device = getattr(engine.executor, "device_name", "cpu")
        app.state.deterministic = settings.deterministic
        app.state.metrics = global_metrics
        engine.start()
        try:
            yield
        finally:
            await engine.aclose()
            logger.info("FluxLLM engine stopped")

    app = FastAPI(
        title="FluxLLM",
        description=(
            "High-throughput LLM inference engine with continuous batching, "
            "block-based KV-cache management, and streaming generation."
        ),
        version=__import__("fluxllm").__version__,
        lifespan=lifespan,
    )
    app.include_router(router)
    if mount_demo(app):
        logger.info("demo page mounted at /demo")
    return app


app = create_app()
