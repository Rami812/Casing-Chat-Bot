#!/usr/bin/env python3
"""
Case Interview ChatBot - Streamlit Version
Optimized for Streamlit Cloud deployment
No AWS dependencies required
"""

import streamlit as st
import google.generativeai as genai
import os
import uuid
import tempfile
from datetime import datetime, timezone
import io
from pathlib import Path
import pickle
import json
from pdfminer.high_level import extract_text
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Case Interview ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .upload-section {
        border: 3px dashed #1f77b4;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        margin: 2rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #1f77b4 0%, #0d6efd 100%);
        color: white;
        margin-left: 20%;
        text-align: right;
    }
    .bot-message {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #2c3e50;
        margin-right: 20%;
        border-left: 4px solid #1f77b4;
    }
    .stage-indicator {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = 'case_introduction'
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False

# Initialize the LLM for CrewAI agents
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY", st.secrets.get("GOOGLE_API_KEY", ""))
    if not api_key:
        st.error("❌ GOOGLE_API_KEY not found! Please set it in your environment variables or Streamlit secrets.")
        st.stop()
    
    my_llm = LLM(
        api_key=api_key,
        model="gemini/gemini-1.5-flash"
    )
    return my_llm

# CrewAI Tools
@tool
def search_case_knowledge(query: str) -> str:
    """Search the case knowledge base for relevant information"""
    return "Search functionality available in chat context"

@tool
def get_current_stage() -> str:
    """Get the current stage of the case interview"""
    return st.session_state.current_stage

@tool
def update_stage(new_stage: str) -> str:
    """Update the current stage of the case interview"""
    st.session_state.current_stage = new_stage
    return f"Stage updated to: {new_stage}"

# CrewAI Case Interview System
class CaseInterviewCrew:
    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base
        self.agents = self._create_agents()

    def _create_agents(self):
        # Case Giver Agent
        case_giver = Agent(
            role="Case Interviewer",
            goal="Conduct a professional case interview, providing case details and evaluating responses",
            backstory="""You are an experienced management consultant and case interviewer. 
            You have conducted hundreds of case interviews and know how to evaluate candidates 
            on their problem-solving, analytical thinking, and communication skills.""",
            verbose=True,
            allow_delegation=False,
            llm=get_llm()
        )

        # Framework Evaluator Agent
        framework_evaluator = Agent(
            role="Framework Evaluation Specialist",
            goal="Evaluate the candidate's use of business frameworks and structured thinking",
            backstory="""You are an expert in business frameworks and structured problem-solving. 
            You evaluate how well candidates apply frameworks like MECE, Porter's 5 Forces, 
            Value Chain Analysis, and other business tools.""",
            verbose=True,
            allow_delegation=False,
            llm=get_llm()
        )

        # Math Evaluator Agent
        math_evaluator = Agent(
            role="Quantitative Analysis Specialist",
            goal="Evaluate the candidate's mathematical reasoning and quantitative analysis",
            backstory="""You are an expert in quantitative analysis and mathematical modeling. 
            You evaluate candidates' ability to perform calculations, interpret data, 
            and make data-driven decisions.""",
            verbose=True,
            allow_delegation=False,
            llm=get_llm()
        )

        # Recommendation Evaluator Agent
        recommendation_evaluator = Agent(
            role="Recommendation Evaluation Specialist",
            goal="Evaluate the quality and feasibility of candidate recommendations",
            backstory="""You are an expert in business strategy and implementation. 
            You evaluate candidates' recommendations for clarity, feasibility, 
            and strategic thinking.""",
            verbose=True,
            allow_delegation=False,
            llm=get_llm()
        )

        return [case_giver, framework_evaluator, math_evaluator, recommendation_evaluator]

    def run_appropriate_task(self, user_input: str, current_stage: str, chat_history: List) -> str:
        """Run the appropriate task based on current stage"""
        if current_stage == "case_introduction":
            return self._run_case_introduction(user_input, chat_history)
        elif current_stage == "clarifying_questions":
            return self._run_clarifying_questions(user_input, chat_history)
        elif current_stage == "framework_evaluation":
            return self._run_framework_evaluation(user_input, chat_history)
        elif current_stage == "math_evaluation":
            return self._run_math_evaluation(user_input, chat_history)
        elif current_stage == "recommendation_evaluation":
            return self._run_recommendation_evaluation(user_input, chat_history)
        else:
            return "I'm ready to help you with your case interview. What would you like to work on?"

    def _run_case_introduction(self, user_input: str, chat_history: List) -> str:
        """Run case introduction task"""
        task = Task(
            description=f"""You are conducting a case interview. The candidate has uploaded a case document.
            Based on the case content and the candidate's input: '{user_input}', provide a professional response.
            Consider the chat history: {chat_history[-3:] if chat_history else []}
            
            Your response should:
            1. Acknowledge the case upload
            2. Provide initial case context if relevant
            3. Guide the candidate to the next step
            4. Be encouraging and professional
            
            Keep your response concise but helpful.""",
            agent=self.agents[0],
            expected_output="A professional response acknowledging the case upload and guiding the candidate to the next step"
        )
        
        # Create a crew with just this task
        crew = Crew(
            agents=[self.agents[0]],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            # Fallback to a simple response
            return f"Welcome to your case interview! I've reviewed your case document. Your input was: '{user_input}'. How would you like to begin?"

    def _run_clarifying_questions(self, user_input: str, chat_history: List) -> str:
        """Run clarifying questions task"""
        task = Task(
            description=f"""The candidate is in the clarifying questions phase. 
            Their input: '{user_input}'
            Chat history: {chat_history[-3:] if chat_history else []}
            
            Help them by:
            1. Identifying what additional information they need
            2. Suggesting specific clarifying questions
            3. Guiding them toward a structured approach
            4. Providing constructive feedback
            
            Be supportive and educational.""",
            agent=self.agents[0],
            expected_output="Specific clarifying questions and guidance for the candidate's structured approach"
        )
        
        # Create a crew with just this task
        crew = Crew(
            agents=[self.agents[0]],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            return f"I can help you with clarifying questions. Based on your input: '{user_input}', what specific information do you need to gather?"

    def _run_framework_evaluation(self, user_input: str, chat_history: List) -> str:
        """Run framework evaluation task"""
        task = Task(
            description=f"""Evaluate the candidate's framework usage. 
            Your input: '{user_input}'
            Chat history: {chat_history[-3:] if chat_history else []}
            
            Assess:
            1. Framework selection appropriateness
            2. Application of the framework
            3. Structured thinking
            4. Areas for improvement
            
            Provide constructive feedback and suggestions.""",
            agent=self.agents[1],
            expected_output="Constructive feedback on framework usage and suggestions for improvement"
        )
        
        # Create a crew with just this task
        crew = Crew(
            agents=[self.agents[1]],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            return f"Let me evaluate your framework usage. Your input: '{user_input}' shows structured thinking. What specific framework are you applying?"

    def _run_math_evaluation(self, user_input: str, chat_history: List) -> str:
        """Run math evaluation task"""
        task = Task(
            description=f"""Evaluate the candidate's quantitative analysis. 
            Your input: '{user_input}'
            Chat history: {chat_history[-3:] if chat_history else []}
            
            Assess:
            1. Mathematical accuracy
            2. Logical reasoning
            3. Data interpretation
            4. Calculation methodology
            
            Provide feedback on their quantitative approach.""",
            agent=self.agents[2],
            expected_output="Feedback on quantitative analysis and mathematical reasoning"
        )
        
        # Create a crew with just this task
        crew = Crew(
            agents=[self.agents[2]],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            return f"Let me evaluate your quantitative analysis. Your input: '{user_input}' shows analytical thinking. What calculations are you performing?"

    def _run_recommendation_evaluation(self, user_input: str, chat_history: List) -> str:
        """Run recommendation evaluation task"""
        task = Task(
            description=f"""Evaluate the candidate's recommendations. 
            Your input: '{user_input}'
            Chat history: {chat_history[-3:] if chat_history else []}
            
            Assess:
            1. Recommendation clarity
            2. Feasibility
            3. Strategic thinking
            4. Implementation considerations
            
            Provide feedback on their recommendations.""",
            agent=self.agents[3],
            expected_output="Feedback on recommendation quality, feasibility, and strategic thinking"
        )
        
        # Create a crew with just this task
        crew = Crew(
            agents=[self.agents[3]],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            return f"Let me evaluate your recommendations. Your input: '{user_input}' shows strategic thinking. What specific recommendations are you making?"

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Case Interview ChatBot</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Session Info")
        
        if st.session_state.session_id:
            st.success(f"**Session ID:** {st.session_state.session_id[:8]}...")
            st.info(f"**Current Stage:** {st.session_state.current_stage.replace('_', ' ').title()}")
            st.info(f"**Messages:** {len(st.session_state.chat_history)}")
            
            if st.button("🔄 Reset Session", type="secondary"):
                st.session_state.session_id = None
                st.session_state.chat_history = []
                st.session_state.current_stage = 'case_introduction'
                st.session_state.knowledge_base = None
                st.session_state.chunks = []
                st.session_state.pdf_uploaded = False
                st.rerun()
        else:
            st.info("No active session")
        
        st.markdown("## 🎯 Interview Stages")
        stages = [
            "Case Introduction",
            "Clarifying Questions", 
            "Framework Evaluation",
            "Math Evaluation",
            "Recommendation Evaluation"
        ]
        
        for i, stage in enumerate(stages):
            stage_key = stage.lower().replace(' ', '_')
            if st.session_state.current_stage == stage_key:
                st.markdown(f"**✅ {stage}**")
            else:
                st.markdown(f"⏳ {stage}")
        
        st.markdown("## 🔧 Settings")
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    if not st.session_state.pdf_uploaded:
        # PDF Upload Section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📄 Upload Your Case Document")
        st.markdown("Start by uploading a PDF case document to begin your interview preparation")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF case document to get started"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Upload & Start Interview", type="primary"):
                with st.spinner("Processing your case document..."):
                    try:
                        # Read file content
                        content = uploaded_file.read()
                        
                        # Extract text from PDF
                        pdf_text = extract_text(io.BytesIO(content))
                        if not pdf_text.strip():
                            st.error("Could not extract text from PDF. Please ensure the PDF contains readable text.")
                            return
                        
                        # Split text into chunks
                        text_splitter = CharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            separator="\n"
                        )
                        chunks = text_splitter.split_text(pdf_text)
                        
                        # Create embeddings and knowledge base
                        try:
                            with st.spinner("Creating AI knowledge base..."):
                                embeddings = HuggingFaceEmbeddings(
                                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                                )
                                knowledge_base = FAISS.from_texts(chunks, embeddings)
                                st.success("✅ Knowledge base created successfully")
                        except Exception as e:
                            st.warning(f"⚠️ Could not create advanced embeddings: {e}")
                            knowledge_base = None
                        
                        # Generate session ID
                        session_id = str(uuid.uuid4())
                        
                        # Update session state
                        st.session_state.session_id = session_id
                        st.session_state.chunks = chunks
                        st.session_state.knowledge_base = knowledge_base
                        st.session_state.pdf_uploaded = True
                        st.session_state.current_stage = 'case_introduction'
                        
                        # Add welcome message
                        welcome_msg = "Welcome to your case interview! I've reviewed your case document. How would you like to begin?"
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': welcome_msg,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        })
                        
                        st.success("🎉 PDF uploaded successfully! You can now start your case interview.")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
                        logger.error(f"Error processing PDF: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Instructions
        st.markdown("## 📋 How It Works")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 1️⃣ **Upload Case**")
            st.markdown("Upload your PDF case document to get started")
        
        with col2:
            st.markdown("### 2️⃣ **AI Analysis**")
            st.markdown("Our AI analyzes your case and creates a knowledge base")
        
        with col3:
            st.markdown("### 3️⃣ **Interview Practice**")
            st.markdown("Practice with our AI interviewer across different stages")
    
    else:
        # Chat Interface
        st.markdown(f'<div class="stage-indicator">🎯 Current Stage: {st.session_state.current_stage.replace("_", " ").title()}</div>', unsafe_allow_html=True)
        
        # Chat messages
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message bot-message">{message["content"]}</div>', unsafe_allow_html=True)
        
        # Chat input
        st.markdown("---")
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Type your message here...",
                key="user_input",
                placeholder="Ask me anything about your case interview..."
            )
        
        with col2:
            if st.button("💬 Send", type="primary", use_container_width=True):
                if user_input.strip():
                    # Add user message to history
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_input,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                    
                    # Get AI response
                    with st.spinner("🤔 Thinking..."):
                        try:
                            crew = CaseInterviewCrew(
                                knowledge_base=st.session_state.knowledge_base
                            )
                            
                            response = crew.run_appropriate_task(
                                user_input,
                                st.session_state.current_stage,
                                st.session_state.chat_history
                            )
                            
                            # Extract the actual text response
                            if hasattr(response, 'raw'):
                                response_text = str(response.raw)
                            elif hasattr(response, 'output'):
                                response_text = str(response.output)
                            else:
                                response_text = str(response)
                            
                            # Clean up the response
                            if response_text.startswith('"') and response_text.endswith('"'):
                                response_text = response_text[1:-1]
                            
                            # Add AI response to history
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': response_text,
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            })
                            
                            # Clear input and rerun
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error getting AI response: {e}")
                            logger.error(f"Error getting AI response: {e}")

if __name__ == "__main__":
    main()
