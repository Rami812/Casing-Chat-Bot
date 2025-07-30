"""
Pydantic schemas for API request and response models
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# Request Models
class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="User message content")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")

class SessionCreateRequest(BaseModel):
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")

# Response Models
class SessionResponse(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    status: str = Field(..., description="Session status")
    message: str = Field(..., description="Status message")
    current_stage: str = Field(..., description="Current interview stage")
    case_uploaded: bool = Field(..., description="Whether case study is uploaded")

class MessageResponse(BaseModel):
    response: str = Field(..., description="AI response content")
    stage: Optional[str] = Field(default=None, description="Current interview stage")
    stage_display: Optional[str] = Field(default=None, description="Human-readable stage name")
    success: bool = Field(..., description="Whether the request was successful")
    session_id: str = Field(..., description="Session identifier")
    error: Optional[str] = Field(default=None, description="Error message if any")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Response timestamp")

class UploadResponse(BaseModel):
    success: bool = Field(..., description="Whether upload was successful")
    message: str = Field(..., description="Upload status message")
    filename: str = Field(..., description="Uploaded filename")
    file_url: Optional[str] = Field(default=None, description="S3 file URL")
    session_ready: bool = Field(..., description="Whether session is ready for interview")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    stage: Optional[str] = Field(default=None, description="Interview stage when sent")

class SessionInfo(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    current_stage: str = Field(..., description="Current interview stage")
    stage_display: str = Field(..., description="Human-readable stage name")
    case_uploaded: bool = Field(..., description="Whether case study is uploaded")
    case_filename: Optional[str] = Field(default=None, description="Case study filename")
    created_at: datetime = Field(..., description="Session creation time")
    analytics: Dict[str, Any] = Field(default_factory=dict, description="Session analytics")

class AnalyticsData(BaseModel):
    stages_completed: List[str] = Field(default_factory=list, description="Completed stages")
    time_per_stage: Dict[str, float] = Field(default_factory=dict, description="Time spent per stage")
    questions_asked: int = Field(default=0, description="Number of questions asked")
    frameworks_submitted: int = Field(default=0, description="Number of frameworks submitted")
    calculations_performed: int = Field(default=0, description="Number of calculations performed")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    aws_services: Dict[str, bool] = Field(..., description="AWS service health")
    timestamp: datetime = Field(..., description="Health check timestamp")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")