from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.app.api.routes import router as api_router
from backend.app.utils.nlp_utils import init_models, clear_all_caches

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_models()
        print("Models initialized successfully!")
    except Exception as e:
        print("Error during startup:", e)
        import traceback
        # await clear_all_caches()
        traceback.print_exc()
        raise e
    yield

app = FastAPI(lifespan=lifespan, title="Video Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")