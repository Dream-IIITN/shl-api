from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from AdvancedSHLRecommender import SHLAdvancedRecommender
import os
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for recommender
recommender = None

# Modern lifespan handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    # Startup logic
    try:
        logger.info("Initializing SHLRecommender...")
        recommender = SHLAdvancedRecommender()
        logger.info("Recommender initialized successfully")
    except Exception as e:
        logger.error(f"Recommender initialization failed: {str(e)}")
        raise RuntimeError("Failed to initialize recommender")
    yield
    # Shutdown logic would go here
    logger.info("Shutting down recommender")

app = FastAPI(
    title="SHL Solution Recommender API",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    text: str
    language: str = "english"
    job_level: str = "Entry Level"
    completion_time: int
    test_type: str

class RecommendationResponse(BaseModel):
    response: str
    sources: list

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendation(request: QueryRequest):
    if not recommender:
        raise HTTPException(status_code=500, detail="Recommender not initialized")
    
    try:
        result = recommender.recommend_solution(
            user_query=request.text,
            user_language=request.language
        )
        
        return {
            "response": result["response"],
            "sources": result.get("sources", [])
        }
        
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": "1.0.0",
        "recommender_ready": recommender is not None
    }

@app.get("/")
async def root():
    return {
        "message": "SHL Recommender API is running",
        "endpoints": {
            "recommend": "POST /recommend",
            "health": "GET /health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))