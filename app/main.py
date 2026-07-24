from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import get_settings
from app.database.db import init_db
from app.routers import forecast, analyze, stress, risk, strategy, pipeline

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title=settings.app_name,
    description="Agentic financial intelligence platform: inflation forecasting, "
                 "financial analysis, stress testing, distress prediction, and strategy generation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "app": settings.app_name, "environment": settings.environment}


app.include_router(forecast.router)
app.include_router(analyze.router)
app.include_router(stress.router)
app.include_router(risk.router)
app.include_router(strategy.router)
app.include_router(pipeline.router)
