# from fastapi import FastAPI, HTTPException, status
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# from AdvancedSHLRecommender import SHLAdvancedRecommender
# import os
# import logging
# from contextlib import asynccontextmanager

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Global variable for recommender
# recommender = None

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Modern lifespan handler for startup/shutdown events"""
#     global recommender
#     # Startup logic
#     try:
#         logger.info("🚀 Initializing SHLRecommender...")
#         recommender = SHLAdvancedRecommender()
#         logger.info("✅ Recommender initialized successfully")
#     except Exception as e:
#         logger.error(f"❌ Recommender initialization failed: {str(e)}")
#         raise RuntimeError("Failed to initialize recommender")
    
#     yield  # App runs here
    
#     # Shutdown logic
#     logger.info("🛑 Shutting down recommender")

# app = FastAPI(
#     title="SHL Solution Recommender API",
#     description="API for recommending SHL assessment solutions",
#     version="1.0.0",
#     lifespan=lifespan
# )

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class QueryRequest(BaseModel):
#     text: str = Field(..., min_length=3, example="Help with numerical reasoning test")
#     language: str = Field(default="english", example="english")
#     job_level: str = Field(default="Entry Level", example="Mid Level")
#     completion_time: int = Field(..., gt=0, example=30)
#     test_type: str = Field(..., example="Numerical Reasoning")

# class RecommendationResponse(BaseModel):
#     response: str
#     sources: list

# @app.post(
#     "/recommend",
#     response_model=RecommendationResponse,
#     status_code=status.HTTP_200_OK,
#     summary="Get SHL solution recommendations",
#     responses={
#         200: {"description": "Successful recommendation"},
#         500: {"description": "Internal server error"}
#     }
# )
# async def get_recommendation(request: QueryRequest):
#     """
#     Get personalized SHL solution recommendations based on:
#     - Test type
#     - Job level
#     - Time requirements
#     - Language preference
#     """
#     if not recommender:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#             detail="Recommender service not ready"
#         )
    
#     try:
#         result = recommender.recommend_solution(
#             user_query=request.text,
#             user_language=request.language
#         )
        
#         return {
#             "response": result["response"],
#             "sources": result.get("sources", [])
#         }
        
#     except Exception as e:
#         logger.error(f"Recommendation error: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error generating recommendation"
#         )

# @app.get(
#     "/health",
#     summary="Check API health status",
#     response_description="API status information"
# )
# async def health_check():
#     """
#     Check if the API is running and ready to accept requests
#     """
#     return {
#         "status": "healthy" if recommender else "initializing",
#         "version": "1.0.0",
#         "ready": recommender is not None
#     }

# @app.get("/", include_in_schema=False)
# async def root():
#     return {
#         "message": "SHL Recommender API is running",
#         "endpoints": {
#             "recommend": {"method": "POST", "path": "/recommend"},
#             "health": {"method": "GET", "path": "/health"}
#         },
#         "documentation": "/docs"
#     }

# def get_port() -> int:
#     """Get port from environment variable or default to 8000"""
#     return int(os.getenv("PORT", "8000"))

# if __name__ == "__main__":
#     import uvicorn
#     port = get_port()
#     logger.info(f"Starting server on port {port}")
#     uvicorn.run(
#         app,
#         host="0.0.0.0",
#         port=port,
#         log_config=None,  # Use default logging config
#         access_log=False  # Disable duplicate access logs
#     )
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from AdvancedSHLRecommender import SHLAdvancedRecommender
import logging

app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommender ONCE
recommender = None

@app.on_event("startup")
async def startup():
    global recommender
    try:
        recommender = SHLAdvancedRecommender()
        logging.info("✅ Recommender initialized")
    except Exception as e:
        logging.error(f"🚨 Recommender init failed: {str(e)}")
        raise

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend")
async def get_recommendation(query: str):
    if not recommender:
        raise HTTPException(502, "Recommender not initialized")
    try:
        return recommender.recommend_solution(query)
    except Exception as e:
        raise HTTPException(500, f"Recommendation failed: {str(e)}")