"""
FastAPI ML Backend - Main Application Entry Point
Run with: uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import dataset, model, training, evaluation, compression, comparison
import os

# Create necessary directories
REQUIRED_DIRS = ["uploads", "models", "results"]
for directory in REQUIRED_DIRS:
    os.makedirs(directory, exist_ok=True)

app = FastAPI(
    title="ML Model Training & Compression API",
    description="Complete ML workflow with training, compression, and evaluation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(dataset.router, prefix="/api/dataset", tags=["Dataset"])
app.include_router(model.router, prefix="/api/model", tags=["Model Selection"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(compression.router, prefix="/api/compression", tags=["Compression"])
app.include_router(comparison.router, prefix="/api/comparison", tags=["Comparison"])

@app.get("/")
async def root():
    return {
        "message": "ML Training & Compression API",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)