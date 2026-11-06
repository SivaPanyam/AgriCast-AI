from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI(
    title="AgriCast-AI Gateway",
    description="Enterprise Multi-Model Price Forecasting, Ensemble Stacking & Market Copilot API",
    version="2.5.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DJANGO_API_URL = "http://127.0.0.1:8000/api/forecast/"

@app.get("/")
def home():
    return {
        "service": "AgriCast-AI Production Gateway",
        "status": "online",
        "version": "2.5.0",
        "supported_models": ["ensemble", "sarima", "holtwinters", "xgboost"]
    }

@app.get("/forecast")
async def get_forecast(
    steps: int = Query(default=12, ge=1, le=36),
    model: str = Query(default="ensemble")
):
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{DJANGO_API_URL}?steps={steps}&model={model}")

        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail="Forecasting Core engine returned an error."
        )
    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Unable to reach Django Core service on port 8000."
        )

@app.post("/forecast")
async def post_forecast(request: Request):
    try:
        body = await request.json()
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(DJANGO_API_URL, json=body)

        response.raise_for_status()
        return response.json()

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail="Forecasting Core engine error."
        )
    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to Django Core forecasting API."
        )
