"""
API Routes for Case Interview Application
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
import uuid
import json
from datetime import datetime

from ..core.config import get_settings
from ..core.aws_manager import AWSManager
from ..core.session_manager import SessionManager
from ..services.case_interview import CaseInterviewService
from ..models.schemas import SessionResponse, MessageRequest, MessageResponse, UploadResponse

settings = get_settings()

# Create routers
case_router = APIRouter()
auth_router = APIRouter()
websocket_router = APIRouter()

# Dependency injection
async def get_aws_manager():
    """Get AWS manager from app state"""
    # This will be injected from main.py app state
    pass

async def get_case_service():
    """Get case interview service from app state"""
    # This will be injected from main.py app state
    pass

# Case Interview Routes
@case_router.post("/session/create", response_model=SessionResponse)
async def create_session(user_id: Optional[str] = None):
    """Create a new case interview session"""
    try:
        from main import app
        case_service = app.state.case_service
        
        session_id = await case_service.create_session(user_id)
        
        return SessionResponse(
            session_id=session_id,
            status="created",
            message="Session created successfully",
            current_stage="case_introduction",
            case_uploaded=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@case_router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    try:
        from main import app
        case_service = app.state.case_service
        
        session_data = await case_service.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "current_stage": session_data.get("current_stage"),
            "stage_display": case_service._get_stage_display_name(session_data.get("current_stage")),
            "case_uploaded": session_data.get("case_processed", False),
            "case_filename": session_data.get("case_filename"),
            "created_at": session_data.get("created_at"),
            "analytics": session_data.get("analytics", {})
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@case_router.post("/session/{session_id}/upload", response_model=UploadResponse)
async def upload_case_study(
    session_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload and process a case study PDF"""
    try:
        from main import app
        case_service = app.state.case_service
        
        # Validate file type
        if file.content_type not in settings.ALLOWED_FILE_TYPES:
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Read file content
        file_content = await file.read()
        
        # Validate file size
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Process the file
        result = await case_service.upload_case_study(session_id, file_content, file.filename)
        
        if result["success"]:
            return UploadResponse(
                success=True,
                message=result["message"],
                filename=file.filename,
                file_url=result.get("file_url"),
                session_ready=result["session_ready"]
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@case_router.post("/session/{session_id}/message", response_model=MessageResponse)
async def send_message(session_id: str, request: MessageRequest):
    """Send a message and get AI response"""
    try:
        from main import app
        case_service = app.state.case_service
        
        result = await case_service.process_message(session_id, request.message)
        
        if result.get("success"):
            return MessageResponse(
                response=result["response"],
                stage=result.get("stage"),
                stage_display=result.get("stage_display"),
                success=True,
                session_id=session_id
            )
        else:
            return MessageResponse(
                response=result.get("response", "An error occurred"),
                error=result.get("error"),
                success=False,
                session_id=session_id
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Message processing failed: {str(e)}")

@case_router.get("/session/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        from main import app
        case_service = app.state.case_service
        
        session_data = await case_service.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "chat_history": session_data.get("chat_history", []),
            "total_messages": len(session_data.get("chat_history", []))
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")

@case_router.get("/session/{session_id}/analytics")
async def get_session_analytics(session_id: str):
    """Get analytics for a session"""
    try:
        from main import app
        case_service = app.state.case_service
        
        analytics = await case_service.get_session_analytics(session_id)
        return {"analytics": analytics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

# WebSocket for real-time communication
@websocket_router.websocket("/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    try:
        from main import app
        case_service = app.state.case_service
        session_manager = app.state.session_manager
        
        # Add connection to session manager
        await session_manager.add_connection(session_id, websocket)
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "message":
                # Process the message
                result = await case_service.process_message(session_id, message_data.get("content", ""))
                
                # Send response back
                response = {
                    "type": "response",
                    "content": result.get("response"),
                    "stage": result.get("stage"),
                    "stage_display": result.get("stage_display"),
                    "success": result.get("success", False),
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send_text(json.dumps(response))
                
            elif message_data.get("type") == "ping":
                # Heartbeat
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        # Remove connection from session manager
        await session_manager.remove_connection(session_id, websocket)
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Error: {str(e)}"
        }))
        await websocket.close()

# Authentication routes (basic)
@auth_router.post("/guest-session")
async def create_guest_session():
    """Create a guest session"""
    guest_id = f"guest_{uuid.uuid4().hex[:8]}"
    
    try:
        from main import app
        case_service = app.state.case_service
        
        session_id = await case_service.create_session(guest_id)
        
        return {
            "guest_id": guest_id,
            "session_id": session_id,
            "message": "Guest session created successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create guest session: {str(e)}")

# Health and status routes
@case_router.get("/health")
async def case_service_health():
    """Check case service health"""
    try:
        from main import app
        aws_manager = app.state.aws_manager
        
        aws_health = await aws_manager.health_check()
        
        return {
            "status": "healthy" if aws_health["overall"] else "degraded",
            "aws_services": aws_health,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }