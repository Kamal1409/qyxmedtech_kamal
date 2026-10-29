"""
FastAPI application entry point for QYX MedTech CASA.

Run with (from backend/):
    uvicorn app.main:app --port 8000
Then open http://localhost:8000 -- the built frontend (frontend/dist) is served
from the same address.  During frontend development run `npm run dev` in
frontend/ instead and open http://localhost:5173.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.config import get_settings
from app.services.casa import get_store
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    get_store().start()          # background worker that runs queued analyses
    yield
    logger.info("Shutting down application")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Computer-aided semen analysis: upload an analyzer video, get motility, "
                "concentration and kinematics. Research use only.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_PREFIX)

# ── Built frontend ────────────────────────────────────────────────────────────
dist = settings.FRONTEND_DIST
if (dist / "index.html").exists():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        """Files from dist/ when they exist, otherwise index.html so client-side
        routes such as /analyses/<id> load the app."""
        if path.startswith(settings.API_PREFIX.strip("/")):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        f = (dist / path).resolve()
        if path and f.is_file() and dist.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(dist / "index.html")
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": f"{settings.APP_NAME} API is running. Build the frontend "
                           "(cd frontend && npm run build) to serve the website here, or see /docs."}
