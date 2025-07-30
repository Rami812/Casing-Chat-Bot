"""
Session Manager for WebSocket connections and real-time communication
"""

import asyncio
import json
from typing import Dict, List, Set
from fastapi import WebSocket
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages WebSocket connections and real-time communication"""
    
    def __init__(self):
        # session_id -> List[WebSocket]
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # WebSocket -> session_id mapping for cleanup
        self.connection_sessions: Dict[WebSocket, str] = {}
        # Track connection metadata
        self.connection_metadata: Dict[WebSocket, Dict] = {}
        
    async def add_connection(self, session_id: str, websocket: WebSocket):
        """Add a WebSocket connection to a session"""
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        
        self.active_connections[session_id].append(websocket)
        self.connection_sessions[websocket] = session_id
        self.connection_metadata[websocket] = {
            "connected_at": datetime.now(timezone.utc),
            "session_id": session_id,
            "last_ping": datetime.now(timezone.utc)
        }
        
        logger.info(f"✅ WebSocket connected to session {session_id}")
        
        # Send welcome message
        await self.send_to_connection(websocket, {
            "type": "connection_established",
            "session_id": session_id,
            "message": "Connected to case interview session",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def remove_connection(self, session_id: str, websocket: WebSocket):
        """Remove a WebSocket connection from a session"""
        try:
            if session_id in self.active_connections:
                if websocket in self.active_connections[session_id]:
                    self.active_connections[session_id].remove(websocket)
                
                # Clean up empty session
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
            
            # Clean up mappings
            if websocket in self.connection_sessions:
                del self.connection_sessions[websocket]
            
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]
                
            logger.info(f"✅ WebSocket disconnected from session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Error removing connection: {str(e)}")
    
    async def send_to_session(self, session_id: str, message: Dict):
        """Send message to all connections in a session"""
        if session_id not in self.active_connections:
            return
        
        message_json = json.dumps(message)
        disconnected_connections = []
        
        for connection in self.active_connections[session_id]:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"❌ Failed to send message to connection: {str(e)}")
                disconnected_connections.append(connection)
        
        # Clean up disconnected connections
        for connection in disconnected_connections:
            await self.remove_connection(session_id, connection)
    
    async def send_to_connection(self, websocket: WebSocket, message: Dict):
        """Send message to a specific connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"❌ Failed to send message to connection: {str(e)}")
            # Clean up the connection
            if websocket in self.connection_sessions:
                session_id = self.connection_sessions[websocket]
                await self.remove_connection(session_id, websocket)
    
    async def broadcast_to_all(self, message: Dict):
        """Broadcast message to all active connections"""
        for session_id in list(self.active_connections.keys()):
            await self.send_to_session(session_id, message)
    
    async def get_session_connections_count(self, session_id: str) -> int:
        """Get number of active connections for a session"""
        return len(self.active_connections.get(session_id, []))
    
    async def get_total_connections_count(self) -> int:
        """Get total number of active connections"""
        return sum(len(connections) for connections in self.active_connections.values())
    
    async def ping_all_connections(self):
        """Send ping to all connections to check if they're alive"""
        ping_message = {
            "type": "ping",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        for session_id in list(self.active_connections.keys()):
            await self.send_to_session(session_id, ping_message)
    
    async def cleanup_stale_connections(self, max_age_minutes: int = 60):
        """Clean up connections that haven't responded to pings"""
        current_time = datetime.now(timezone.utc)
        stale_connections = []
        
        for websocket, metadata in self.connection_metadata.items():
            last_ping = metadata.get("last_ping", metadata["connected_at"])
            age_minutes = (current_time - last_ping).total_seconds() / 60
            
            if age_minutes > max_age_minutes:
                stale_connections.append((websocket, metadata["session_id"]))
        
        # Remove stale connections
        for websocket, session_id in stale_connections:
            logger.info(f"🧹 Cleaning up stale connection for session {session_id}")
            await self.remove_connection(session_id, websocket)
    
    def get_connection_stats(self) -> Dict:
        """Get statistics about active connections"""
        total_connections = sum(len(connections) for connections in self.active_connections.values())
        
        return {
            "total_sessions": len(self.active_connections),
            "total_connections": total_connections,
            "sessions_with_connections": [
                {
                    "session_id": session_id,
                    "connection_count": len(connections),
                    "connected_at": [
                        self.connection_metadata.get(conn, {}).get("connected_at", "unknown").isoformat()
                        if hasattr(self.connection_metadata.get(conn, {}).get("connected_at", ""), "isoformat")
                        else "unknown"
                        for conn in connections
                    ]
                }
                for session_id, connections in self.active_connections.items()
            ]
        }