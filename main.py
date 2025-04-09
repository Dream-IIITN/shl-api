from fastapi import FastAPI, HTTPException, Depends, Header, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from AdvancedSHLRecommender import SHLAdvancedRecommender
import os
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SHL Solution Recommender API",
    description="API for recommending SHL assessments with AI",
    version="1.0.0"
)

# API Key Security Configuration
API_KEY_NAME = "X-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key",
            headers={"WWW-Authenticate": "API-Key"}
        )
    return api_key

# Configure CORS - Be more restrictive in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-KEY", "Content-Type"],
    expose_headers=["X-API-KEY"]
)

# Initialize recommender once at startup
@app.on_event("startup")
def startup_event():
    global recommender
    try:
        logger.info("Initializing SHLRecommender...")
        recommender = SHLAdvancedRecommender()
        logger.info("Recommender initialized successfully")
    except Exception as e:
        logger.error(f"Recommender initialization failed: {str(e)}")
        raise RuntimeError("Failed to initialize recommender")

class QueryRequest(BaseModel):
    text: str
    language: str = "english"
    job_level: str = "Entry Level"
    completion_time: Optional[int] = None
    test_type: Optional[str] = None

class RecommendationResponse(BaseModel):
    response: str
    sources: list

class HealthResponse(BaseModel):
    status: str
    version: str

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendation(
    request: QueryRequest,
    api_key: str = Depends(validate_api_key)
):
    """Get assessment recommendations with API key authentication"""
    try:
        result = recommender.recommend_solution(
            user_query=request.text,
            user_language=request.language
        )
        
        return {
            "response": result["response"],
            "sources": result["sources"] if result["sources"] else []
        }
        
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
            headers={"X-Error": str(e)}
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "SHL Recommender API is running",
        "documentation": "/docs",
        "endpoints": {
            "recommend": {"method": "POST", "path": "/recommend"},
            "health": {"method": "GET", "path": "/health"}
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )