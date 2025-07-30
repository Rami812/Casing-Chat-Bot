"""
Case Interview Service - Migrated from Chainlit to FastAPI
Integrates with AWS for storage and maintains CrewAI multi-agent functionality
"""

import os
import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import io
from pdfminer.high_level import extract_text
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import TensorflowHubEmbeddings
from langchain.vectorstores import FAISS
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import json
import logging

from ..core.config import get_settings
from ..core.aws_manager import AWSManager

logger = logging.getLogger(__name__)

class CaseInterviewService:
    """Service for managing case interview sessions with AWS integration"""
    
    def __init__(self, aws_manager: AWSManager):
        self.aws_manager = aws_manager
        self.settings = get_settings()
        self.active_sessions: Dict[str, Dict] = {}
        self.knowledge_bases: Dict[str, Any] = {}
        
    def _get_llm(self):
        """Initialize the LLM for CrewAI agents"""
        api_key = self.settings.GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        return LLM(
            api_key=api_key,
            model="gemini/gemini-1.5-flash"
        )
    
    async def create_session(self, user_id: Optional[str] = None) -> str:
        """Create a new case interview session"""
        session_id = str(uuid.uuid4())
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "current_stage": "case_introduction",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "case_file_key": None,
            "knowledge_base_key": None,
            "chat_history": [],
            "analytics": {
                "stages_completed": [],
                "time_per_stage": {},
                "questions_asked": 0,
                "frameworks_submitted": 0,
                "calculations_performed": 0
            }
        }
        
        # Store in memory for quick access
        self.active_sessions[session_id] = session_data
        
        # Persist to DynamoDB
        await self.aws_manager.save_session_data(session_id, session_data)
        
        logger.info(f"✅ Created new session: {session_id}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data, from memory or DynamoDB"""
        # Try memory first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Fallback to DynamoDB
        session_data = await self.aws_manager.get_session_data(session_id)
        if session_data:
            self.active_sessions[session_id] = session_data
        
        return session_data
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session data"""
        session_data = await self.get_session(session_id)
        if not session_data:
            return False
        
        session_data.update(updates)
        session_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        
        # Update memory
        self.active_sessions[session_id] = session_data
        
        # Persist to DynamoDB
        return await self.aws_manager.save_session_data(session_id, session_data)
    
    async def upload_case_study(self, session_id: str, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Upload and process a case study PDF"""
        try:
            # Generate unique file key
            file_key = f"{self.settings.S3_PDF_PREFIX}{session_id}/{filename}"
            
            # Upload to S3
            file_url = await self.aws_manager.upload_file_to_s3(
                file_content, file_key, "application/pdf"
            )
            
            # Extract text from PDF
            text = extract_text(io.BytesIO(file_content))
            
            # Create knowledge base
            knowledge_base = await self._create_knowledge_base(text, session_id)
            
            # Update session
            await self.update_session(session_id, {
                "case_file_key": file_key,
                "case_file_url": file_url,
                "case_filename": filename,
                "knowledge_base_key": f"kb_{session_id}",
                "case_processed": True
            })
            
            # Store knowledge base in memory
            self.knowledge_bases[session_id] = knowledge_base
            
            # Log analytics
            await self.aws_manager.save_analytics_data(
                session_id, "case_uploaded", 
                {"filename": filename, "file_size": len(file_content)}
            )
            
            return {
                "success": True,
                "message": f"Case study '{filename}' processed successfully!",
                "file_url": file_url,
                "session_ready": True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to process case study: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to process case study: {str(e)}",
                "session_ready": False
            }
    
    async def _create_knowledge_base(self, text: str, session_id: str) -> Any:
        """Create FAISS knowledge base from extracted text"""
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings and knowledge base
        embeddings = TensorflowHubEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        return knowledge_base
    
    def _create_search_tool(self, session_id: str):
        """Create knowledge base search tool for specific session"""
        @tool
        def search_case_knowledge(query: str) -> str:
            """Search the uploaded case document for relevant information."""
            if session_id in self.knowledge_bases:
                knowledge_base = self.knowledge_bases[session_id]
                docs = knowledge_base.similarity_search(query, k=self.settings.SIMILARITY_SEARCH_K)
                return "\n".join([doc.page_content for doc in docs])
            return "No knowledge base available"
        
        return search_case_knowledge
    
    def _create_stage_tools(self, session_id: str):
        """Create stage management tools for specific session"""
        @tool
        def get_current_stage() -> str:
            """Get the current stage of the case interview."""
            session_data = self.active_sessions.get(session_id, {})
            return session_data.get('current_stage', 'case_introduction')
        
        @tool
        def update_stage(new_stage: str) -> str:
            """Update the current stage of the case interview."""
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['current_stage'] = new_stage
                # Async update would be better, but tools need to be sync
                return f"Stage updated to: {new_stage}"
            return "Session not found"
        
        return get_current_stage, update_stage
    
    def _create_case_interview_crew(self, session_id: str):
        """Create CrewAI system for specific session"""
        llm = self._get_llm()
        search_tool = self._create_search_tool(session_id)
        get_stage, update_stage = self._create_stage_tools(session_id)
        
        # Case Giver Agent
        case_giver = Agent(
            role="Case Giver",
            goal="Present case scenarios and validate clarifying questions from students",
            backstory="""You are a McKinsey consultant with extensive business knowledge and industry expertise. 
            You specialize in presenting case studies and evaluating the quality of clarifying questions. 
            You ensure students ask strategic, framework-oriented questions rather than diving into specific numbers.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[search_tool, get_stage, update_stage]
        )
        
        # Framework Checker Agent
        framework_checker = Agent(
            role="Framework Checker",
            goal="Evaluate and improve student frameworks for case solving",
            backstory="""You are a consulting case prep instructor who has conducted hundreds of case interviews. 
            You excel at identifying whether frameworks are MECE (Mutually Exclusive, Collectively Exhaustive) 
            and provide guidance on industry-specific terminology and structure.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[search_tool, get_stage]
        )
        
        # Business/Math Response Checker Agent
        math_checker = Agent(
            role="Business Math Checker",
            goal="Verify mathematical calculations and business logic",
            backstory="""You are a consultant who specializes in quantitative analysis and business mathematics. 
            You know arithmetic shortcuts, proper rounding techniques, and can explain complex business concepts 
            using simple analogies like pizza shops or retail stores.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[search_tool, get_stage]
        )
        
        # Recommendation and Summary Checker Agent
        recommendation_checker = Agent(
            role="Recommendation Checker",
            goal="Evaluate final recommendations and summaries",
            backstory="""You are a senior consultant who has presented to hundreds of CEOs and C-suite executives. 
            You ensure recommendations are actionable, properly formatted, include risk mitigation strategies, 
            and end with clear next steps.""",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            tools=[search_tool, get_stage]
        )
        
        return {
            'case_giver': case_giver,
            'framework_checker': framework_checker,
            'math_checker': math_checker,
            'recommendation_checker': recommendation_checker
        }
    
    async def process_message(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """Process user message and return AI response"""
        try:
            session_data = await self.get_session(session_id)
            if not session_data:
                return {"error": "Session not found", "response": None}
            
            if not session_data.get("case_processed"):
                return {
                    "error": "No case study uploaded", 
                    "response": "Please upload a case study PDF before starting the interview."
                }
            
            # Create CrewAI system for this session
            agents = self._create_case_interview_crew(session_id)
            current_stage = session_data.get('current_stage', 'case_introduction')
            
            # Determine appropriate task and agent
            task_info = self._determine_task(user_message, current_stage, agents)
            
            # Create and execute task
            crew = Crew(
                agents=[task_info['agent']],
                tasks=[task_info['task']],
                verbose=True,
                process=Process.sequential
            )
            
            result = crew.kickoff()
            
            # Update session with new message and response
            chat_history = session_data.get('chat_history', [])
            chat_history.extend([
                {"role": "user", "content": user_message, "timestamp": datetime.now(timezone.utc).isoformat()},
                {"role": "assistant", "content": result.raw, "stage": current_stage, "timestamp": datetime.now(timezone.utc).isoformat()}
            ])
            
            # Update analytics
            analytics = session_data.get('analytics', {})
            analytics['questions_asked'] = analytics.get('questions_asked', 0) + 1
            
            await self.update_session(session_id, {
                'chat_history': chat_history,
                'analytics': analytics
            })
            
            # Log analytics
            await self.aws_manager.save_analytics_data(
                session_id, "message_processed",
                {"stage": current_stage, "user_message_length": len(user_message)}
            )
            
            return {
                "response": result.raw,
                "stage": session_data.get('current_stage'),
                "stage_display": self._get_stage_display_name(session_data.get('current_stage')),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to process message: {str(e)}")
            return {
                "error": f"Failed to process message: {str(e)}",
                "response": "Sorry, I encountered an error processing your message. Please try again.",
                "success": False
            }
    
    def _determine_task(self, user_input: str, current_stage: str, agents: Dict) -> Dict[str, Any]:
        """Determine which task to run based on current stage and user input"""
        # Stage transition logic (same as original)
        new_stage = current_stage
        
        if current_stage == 'case_introduction':
            if 'framework' in user_input.lower() or 'approach' in user_input.lower():
                new_stage = 'framework_evaluation'
            elif any(question_word in user_input.lower() for question_word in ['who', 'what', 'where', 'when', 'why', 'how']):
                new_stage = 'clarifying_questions'
        elif current_stage == 'clarifying_questions':
            if 'framework' in user_input.lower() or 'approach' in user_input.lower():
                new_stage = 'framework_evaluation'
        elif current_stage == 'framework_evaluation':
            if any(calc_word in user_input.lower() for calc_word in ['calculate', 'math', 'number', '$', '%']):
                new_stage = 'math_evaluation'
        elif current_stage == 'math_evaluation':
            if any(rec_word in user_input.lower() for rec_word in ['recommend', 'suggestion', 'conclusion', 'summary']):
                new_stage = 'recommendation_evaluation'
        
        # Update stage if changed
        if new_stage != current_stage:
            asyncio.create_task(self.update_session(self.current_session_id, {'current_stage': new_stage}))
        
        # Create appropriate task
        task_descriptions = {
            'case_introduction': self._get_case_introduction_task_desc(),
            'clarifying_questions': self._get_clarifying_questions_task_desc(),
            'framework_evaluation': self._get_framework_evaluation_task_desc(),
            'math_evaluation': self._get_math_evaluation_task_desc(),
            'recommendation_evaluation': self._get_recommendation_evaluation_task_desc()
        }
        
        agent_mapping = {
            'case_introduction': agents['case_giver'],
            'clarifying_questions': agents['case_giver'],
            'framework_evaluation': agents['framework_checker'],
            'math_evaluation': agents['math_checker'],
            'recommendation_evaluation': agents['recommendation_checker']
        }
        
        task = Task(
            description=task_descriptions[new_stage] + f"\n\nUser Input: {user_input}",
            agent=agent_mapping[new_stage],
            expected_output="Detailed response with specific feedback and guidance"
        )
        
        return {'task': task, 'agent': agent_mapping[new_stage]}
    
    def _get_case_introduction_task_desc(self) -> str:
        return """Present the case scenario from the uploaded document to the student. 
        Provide a clear, structured case prompt that includes:
        1. The business situation
        2. The key challenge or question
        3. Any initial context needed
        
        Keep the presentation engaging and professional, as if you're conducting a real McKinsey interview."""
    
    def _get_clarifying_questions_task_desc(self) -> str:
        return """Evaluate the student's clarifying questions based on these criteria:
        1. Are they strategic and framework-oriented (not number-focused)?
        2. Do they help understand the business context?
        3. Are they relevant to the industry/situation?
        
        Good examples:
        - "Who is our target audience?"
        - "How soon do we need to complete this project?"
        - "Have we been profitable historically?"
        
        If questions are too specific or number-based, provide hints like:
        - "What might be important to consider for this specific industry?"
        - "Think about the broader business context first"
        
        Respond with feedback and either answer good questions or redirect poor ones."""
    
    def _get_framework_evaluation_task_desc(self) -> str:
        return """Evaluate the student's framework against these criteria:
        1. Is it MECE (Mutually Exclusive, Collectively Exhaustive)?
        2. Does it have at least 3 main factors?
        3. Are there 3+ questions for each factor?
        4. Does it match the solution approach reasonably?
        5. Does it use appropriate business terminology for the industry?
        
        Example good frameworks:
        - Profitability framework (Revenue, Costs, Market factors)
        - Market Entry framework (Market, Competition, Company capabilities)
        
        Provide specific feedback on what's missing or could be improved. 
        Use industry-specific language (e.g., "number of seats" for theaters, not just "quantity")."""
    
    def _get_math_evaluation_task_desc(self) -> str:
        return """Check the student's mathematical work for:
        1. Correct numbers and units usage
        2. Proper rounding techniques (round one up, one down in multiplication; both down in division)
        3. Step-by-step solution approach
        4. Logical flow of calculations
        
        If the student asks for explanations, use simple analogies:
        - Cost of Goods Sold → pizza ingredients cost
        - Economies of scale → bulk buying discounts
        - Market share → slice of the total pizza
        
        Provide hints for errors and detailed explanations when requested."""
    
    def _get_recommendation_evaluation_task_desc(self) -> str:
        return """Evaluate the student's final recommendation for:
        1. Use of correct final numbers from math section
        2. Professional, CEO-appropriate language and tone
        3. Inclusion of risk mitigation strategies
        4. Clear next steps at the end
        5. Business terminology usage
        
        The recommendation should sound like it's being presented to a CEO, with:
        - Executive summary of findings
        - Key recommendations with supporting rationale
        - Risk mitigation strategies
        - Specific, actionable next steps
        
        Provide feedback on tone, structure, and completeness."""
    
    def _get_stage_display_name(self, stage: str) -> str:
        """Get display name for stage"""
        stage_names = {
            'case_introduction': 'Stage 1 - Case Introduction',
            'clarifying_questions': 'Stage 2 - Clarifying Questions',
            'framework_evaluation': 'Stage 3 - Framework Development',
            'math_evaluation': 'Stage 4 - Quantitative Analysis',
            'recommendation_evaluation': 'Stage 5 - Recommendations'
        }
        return stage_names.get(stage, 'Unknown Stage')
    
    async def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics data for a session"""
        session_data = await self.get_session(session_id)
        if not session_data:
            return {}
        
        return session_data.get('analytics', {})