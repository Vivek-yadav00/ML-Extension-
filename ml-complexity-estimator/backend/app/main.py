from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import analyze

app = FastAPI(title="ML Complexity Estimator API")

# Allow CORS for Extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to extension ID
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router, prefix="/api", tags=["analysis"])

@app.get("/")
def read_root():
    return {"message": "Service is running"}
