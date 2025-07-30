"""
Configuration management for the Case Interview application
Handles environment variables and AWS settings
"""

import os
from functools import lru_cache
from typing import Optional
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings with AWS integration"""
    
    # Basic app configuration
    APP_NAME: str = "AI Case Interview Coach"
    VERSION: str = "2.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key", env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # AI/LLM Configuration
    GOOGLE_API_KEY: str = Field(env="GOOGLE_API_KEY")
    GROQ_API_KEY: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    
    # AWS Configuration (Free Tier)
    AWS_ACCESS_KEY_ID: str = Field(env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = Field(env="AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = Field(default="us-east-1", env="AWS_REGION")
    
    # AWS S3 Configuration
    S3_BUCKET_NAME: str = Field(default="case-interview-files", env="S3_BUCKET_NAME")
    S3_PDF_PREFIX: str = Field(default="case-studies/", env="S3_PDF_PREFIX")
    S3_KNOWLEDGE_BASE_PREFIX: str = Field(default="knowledge-bases/", env="S3_KNOWLEDGE_BASE_PREFIX")
    
    # AWS DynamoDB Configuration
    DYNAMODB_TABLE_SESSIONS: str = Field(default="case-interview-sessions", env="DYNAMODB_TABLE_SESSIONS")
    DYNAMODB_TABLE_USERS: str = Field(default="case-interview-users", env="DYNAMODB_TABLE_USERS")
    DYNAMODB_TABLE_ANALYTICS: str = Field(default="case-interview-analytics", env="DYNAMODB_TABLE_ANALYTICS")
    
    # Redis Configuration (for caching and sessions)
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # File upload limits
    MAX_FILE_SIZE: int = Field(default=20 * 1024 * 1024, env="MAX_FILE_SIZE")  # 20MB
    ALLOWED_FILE_TYPES: list = ["application/pdf"]
    
    # Knowledge base configuration
    CHUNK_SIZE: int = Field(default=1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=200, env="CHUNK_OVERLAP")
    SIMILARITY_SEARCH_K: int = Field(default=3, env="SIMILARITY_SEARCH_K")
    
    # WebSocket configuration
    WS_MESSAGE_QUEUE_SIZE: int = Field(default=100, env="WS_MESSAGE_QUEUE_SIZE")
    WS_HEARTBEAT_INTERVAL: int = Field(default=30, env="WS_HEARTBEAT_INTERVAL")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()