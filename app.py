from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from model import load_weights, save_weights, train_from_public_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "model_weights.pkl"
LEGACY_WEIGHTS_PATH = ROOT / "model_weights.json"
PUBLIC_DATA_PATH = ROOT / "relevant_priors_public.json"

_model = None
_model_lock = threading.Lock()


def _load_or_train_model():
    global _model
    if WEIGHTS_PATH.exists():
        log.info("Loading model from %s", WEIGHTS_PATH)
        _model = load_weights(WEIGHTS_PATH)
    elif LEGACY_WEIGHTS_PATH.exists():
        log.info("Loading model from %s", LEGACY_WEIGHTS_PATH)
        _model = load_weights(LEGACY_WEIGHTS_PATH)
    elif PUBLIC_DATA_PATH.exists():
        log.info("Training model from public data %s", PUBLIC_DATA_PATH)
        trained = train_from_public_json(PUBLIC_DATA_PATH)
        save_weights(trained, WEIGHTS_PATH)
        _model = trained
        log.info("Model trained and saved to %s", WEIGHTS_PATH)
    else:
        raise RuntimeError("No model weights or training data found.")
    log.info("Model ready — mode=%s threshold=%.2f", _model.mode, _model.threshold)


def _get_model():
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            _load_or_train_model()
    return _model


@asynccontextmanager
async def lifespan(application: FastAPI):
    _load_or_train_model()
    yield


app = FastAPI(title="Relevant Priors Endpoint", version="1.0.0", lifespan=lifespan)


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    try:
        payload: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    challenge_id = payload.get("challenge_id")
    schema_version = payload.get("schema_version")

    if challenge_id != "relevant-priors-v1":
        raise HTTPException(status_code=400, detail="challenge_id must be 'relevant-priors-v1'.")
    if schema_version != 1:
        raise HTTPException(status_code=400, detail="schema_version must be 1.")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise HTTPException(status_code=400, detail="cases must be a list.")

    case_count = len(cases)
    prior_count = sum(len(c.get("prior_studies", [])) for c in cases)
    log.info("[%s] request cases=%d priors=%d", request_id, case_count, prior_count)

    model = _get_model()
    result = model.predict(payload)

    elapsed = time.perf_counter() - t0
    log.info("[%s] done predictions=%d elapsed=%.3fs", request_id, len(result["predictions"]), elapsed)

    return JSONResponse(content=result)
