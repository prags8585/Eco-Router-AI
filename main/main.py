
import os
import time
import asyncio
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN")
TIMEOUT_S = float(os.getenv("HF_TIMEOUT_S", "30"))

LANE_MODELS = {
    "fast":     {"repo": "google/flan-t5-small",          "task": "text2text-generation"},
    "balanced": {"repo": "google/flan-t5-large",          "task": "text2text-generation"},
    "deep":     {"repo": "mistralai/Mistral-7B-Instruct-v0.2", "task": "text-generation"},
}
HF_BASE = "https://api-inference.huggingface.co/models"

DEFAULT_PARAMS = {
    "text2text-generation": {"max_new_tokens": 128, "temperature": 0.2},
    "text-generation":      {"max_new_tokens": 128, "temperature": 0.7, "return_full_text": False},
}

app = FastAPI(title="AI Router Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RouteRequest(BaseModel):
    prompt: str = Field(..., description="User prompt from Playground")
    lane_selector: Optional[str] = Field(default="Auto (Default)")
    analyzer_intent: Optional[str] = None
    analyzer_difficulty: Optional[str] = None
    analyzer_min_lane: Optional[str] = None

class ModelResult(BaseModel):
    model_id: str
    lane: str
    latency_ms: int
    text: str
    error: Optional[str] = None

class RouteResponse(BaseModel):
    prompt: str
    chosen_lane: str
    results: Dict[str, ModelResult]

def _endpoint_for(repo: str) -> str:
    return f"{HF_BASE}/{repo}"

async def _call_hf(repo: str, task: str, prompt: str) -> Dict[str, Any]:
    url = _endpoint_for(repo)
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
    payload = {"inputs": prompt, "parameters": DEFAULT_PARAMS.get(task, {"max_new_tokens": 128})}
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        r = await client.post(url, headers=headers, json=payload)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    try:
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            text = data[0].get("generated_text", "") or data[0].get("summary_text", "")
        elif isinstance(data, dict) and "generated_text" in data:
            text = data["generated_text"]
        elif isinstance(data, str):
            text = data
        else:
            text = str(data)[:800]
        return {"text": text, "latency_ms": latency_ms}
    except httpx.HTTPStatusError:
        return {"text": "", "latency_ms": latency_ms, "error": r.text[:300]}
    except Exception as e:
        return {"text": "", "latency_ms": latency_ms, "error": str(e)[:300]}

def _auto_lane(req: RouteRequest) -> str:
    if req.analyzer_min_lane in {"Fast", "Balanced", "Deep"}:
        return req.analyzer_min_lane.lower()
    n = len(req.prompt.split())
    if n <= 10:
        return "fast"
    if n >= 100 or (req.analyzer_difficulty or "").lower() == "high":
        return "deep"
    return "balanced"

@app.post("/api/route", response_model=RouteResponse)
async def route(req: RouteRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt")
    chosen_lane = (req.lane_selector.lower().split()[0] if req.lane_selector else "auto")
    if chosen_lane == "auto":
        chosen_lane = _auto_lane(req)

    async def run_lane(lane: str):
        cfg = LANE_MODELS[lane]
        out = await _call_hf(cfg["repo"], cfg["task"], req.prompt)
        return lane, ModelResult(
            model_id=cfg["repo"],
            lane=lane,
            latency_ms=out["latency_ms"],
            text=out.get("text", ""),
            error=out.get("error"),
        )

    lanes = ["fast", "balanced", "deep"]
    results_pairs = await asyncio.gather(*[run_lane(l) for l in lanes])
    results = {k: v for k, v in results_pairs}
    return RouteResponse(prompt=req.prompt, chosen_lane=chosen_lane, results=results)

@app.get("/healthz")
def healthz():
    return {"ok": True}
