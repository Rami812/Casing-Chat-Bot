"""
FastAPI-based Case Interview Practice Application
Improved version with AWS integration and modern web architecture
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from dotenv import load_dotenv

# Import our custom modules
from app.core.config import get_settings
from app.core.aws_manager import AWSManager
from app.core.session_manager import SessionManager
from app.api.routes import case_router, auth_router, websocket_router
from app.services.case_interview import CaseInterviewService
from app.core.database import init_db

# Load environment variables
load_dotenv()

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown"""
    # Startup
    print("🚀 Starting Case Interview API...")
    
    # Initialize database
    await init_db()
    
    # Initialize AWS services
    aws_manager = AWSManager()
    await aws_manager.initialize()
    
    # Store in app state
    app.state.aws_manager = aws_manager
    app.state.session_manager = SessionManager()
    app.state.case_service = CaseInterviewService(aws_manager)
    
    print("✅ Application startup complete!")
    yield
    
    # Shutdown
    print("🛑 Shutting down Case Interview API...")
    # Clean up resources if needed
    print("✅ Shutdown complete!")

# Create FastAPI app
app = FastAPI(
    title="AI Case Interview Coach",
    description="Multi-agent system for comprehensive case interview preparation with AWS cloud integration",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
app.include_router(case_router, prefix="/api/case", tags=["case-interview"])
app.include_router(websocket_router, prefix="/ws", tags=["websocket"])

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend application"""
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "aws_connected": True,  # Will be dynamically checked
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.get("/api/info")
async def get_api_info():
    """Get API information and configuration"""
    return {
        "name": "AI Case Interview Coach API",
        "version": "2.0.0",
        "description": "FastAPI backend with AWS integration for case interview practice",
        "features": [
            "Multi-agent AI coaching system",
            "PDF case study processing",
            "Real-time interview sessions",
            "AWS cloud storage",
            "Session persistence",
            "Performance analytics"
        ],
        "aws_services": [
            "S3 for file storage",
            "DynamoDB for session data",
            "Lambda for processing",
            "CloudWatch for monitoring"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )