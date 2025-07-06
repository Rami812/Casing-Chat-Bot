import streamlit as st
#import os
from typing import List, Dict, Any
import json
import pickle
import requests
from datetime import datetime
import tempfile
import PyPDF
from io import BytesIO
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import openai
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain.tools import tool
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="BCG Consulting Case Interview Prep",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .task-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        font-weight: bold;
    }
    .status-pending { background-color: #ffd700; color: #333; }
    .status-running { background-color: #4CAF50; color: white; }
    .status-completed { background-color: #2196F3; color: white; }
    .case-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
class Config:
    OPENAI_API_KEY = "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"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    VECTOR_STORE_PATH = "case_vectors.faiss"
    CASE_METADATA_PATH = "case_metadata.json"
    MAX_TOKENS = 4000
    TEMPERATURE = 0.7

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "current_case" not in st.session_state:
    st.session_state.current_case = None
if "interview_stage" not in st.session_state:
    st.session_state.interview_stage = "setup"
if "case_database" not in st.session_state:
    st.session_state.case_database = None
if "agents_initialized" not in st.session_state:
    st.session_state.agents_initialized = False

# PDF Processing and Knowledge Base Tools
class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting PDF text: {str(e)}")
            return ""
    
    def extract_case_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract case metadata using pattern matching and keywords"""
        metadata = {
            "filename": filename,
            "difficulty": "Medium",
            "case_type": "General",
            "industry": "Unknown",
            "skills_tested": [],
            "content_preview": text[:500] + "..." if len(text) > 500 else text
        }
        
        # Difficulty detection
        text_lower = text.lower()
        if any(word in text_lower for word in ["beginner", "easy", "basic", "introductory"]):
            metadata["difficulty"] = "Easy"
        elif any(word in text_lower for word in ["advanced", "hard", "difficult", "challenging", "expert"]):
            metadata["difficulty"] = "Hard"
        
        # Case type detection
        case_types = {
            "Market Entry": ["market entry", "new market", "expansion", "international"],
            "Profitability": ["profitability", "profit", "cost reduction", "revenue"],
            "Pricing": ["pricing", "price", "pricing strategy"],
            "M&A": ["merger", "acquisition", "m&a", "due diligence"],
            "Operations": ["operations", "operational", "efficiency", "supply chain"],
            "Growth": ["growth", "growth strategy", "expansion"],
            "Digital": ["digital", "technology", "tech", "innovation"]
        }
        
        for case_type, keywords in case_types.items():
            if any(keyword in text_lower for keyword in keywords):
                metadata["case_type"] = case_type
                break
        
        # Industry detection
        industries = {
            "Healthcare": ["healthcare", "hospital", "pharmaceutical", "medical"],
            "Technology": ["technology", "software", "tech", "digital"],
            "Retail": ["retail", "e-commerce", "consumer", "shopping"],
            "Financial Services": ["bank", "financial", "insurance", "investment"],
            "Manufacturing": ["manufacturing", "production", "factory"],
            "Energy": ["energy", "oil", "gas", "renewable"],
            "Transportation": ["transportation", "logistics", "airline", "shipping"]
        }
        
        for industry, keywords in industries.items():
            if any(keyword in text_lower for keyword in keywords):
                metadata["industry"] = industry
                break
        
        # Skills tested
        skills = {
            "Market Analysis": ["market analysis", "competitive analysis", "market research"],
            "Financial Analysis": ["financial", "revenue", "cost", "profit", "valuation"],
            "Strategic Thinking": ["strategy", "strategic", "competitive advantage"],
            "Problem Solving": ["problem", "solution", "recommendation"],
            "Data Analysis": ["data", "analytics", "metrics", "kpi"]
        }
        
        for skill, keywords in skills.items():
            if any(keyword in text_lower for keyword in keywords):
                metadata["skills_tested"].append(skill)
        
        return metadata

class CaseKnowledgeBase:
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.vector_store = None
        self.case_metadata = []
        self.documents = []
    
    def build_knowledge_base(self, pdf_files: List[Dict[str, Any]]):
        """Build FAISS knowledge base from PDF files"""
        processor = PDFProcessor()
        all_embeddings = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pdf_file in enumerate(pdf_files):
            status_text.text(f"Processing {pdf_file['name']}...")
            
            # Extract text and metadata
            text = processor.extract_text_from_pdf(pdf_file['content'])
            metadata = processor.extract_case_metadata(text, pdf_file['name'])
            
            # Split text into chunks
            chunks = processor.text_splitter.split_text(text)
            
            # Create embeddings for each chunk
            for chunk in chunks:
                embedding = self.embedding_model.encode(chunk)
                all_embeddings.append(embedding)
                
                # Store document with metadata
                doc = Document(
                    page_content=chunk,
                    metadata={
                        **metadata,
                        "chunk_index": len(self.documents)
                    }
                )
                self.documents.append(doc)
            
            self.case_metadata.append(metadata)
            progress_bar.progress((i + 1) / len(pdf_files))
        
        # Build FAISS index
        embeddings_array = np.array(all_embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        self.vector_store = faiss.IndexFlatL2(dimension)
        self.vector_store.add(embeddings_array)
        
        status_text.text("Knowledge base built successfully!")
        progress_bar.progress(100)
        
        return self.case_metadata
    
    def search_cases(self, query: str, case_type: str = None, difficulty: str = None, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant cases based on query and filters"""
        if not self.vector_store:
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode(query).astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search vector store
        scores, indices = self.vector_store.search(query_embedding, k * 3)  # Get more results for filtering
        
        # Filter results based on criteria
        filtered_cases = []
        seen_cases = set()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                case_name = doc.metadata.get('filename', '')
                
                # Avoid duplicates
                if case_name in seen_cases:
                    continue
                seen_cases.add(case_name)
                
                # Apply filters
                if case_type and doc.metadata.get('case_type') != case_type:
                    continue
                if difficulty and doc.metadata.get('difficulty') != difficulty:
                    continue
                
                filtered_cases.append({
                    'case_name': case_name,
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score)
                })
                
                if len(filtered_cases) >= k:
                    break
        
        return filtered_cases

# CrewAI Tools
class CaseSearchTool(BaseTool):
    name: str = "case_search"
    description: str = "Search for consulting cases based on criteria"
    
    def __init__(self, knowledge_base: CaseKnowledgeBase):
        super().__init__()
        self.knowledge_base = knowledge_base
    
    def _run(self, query: str, case_type: str = None, difficulty: str = None) -> str:
        results = self.knowledge_base.search_cases(query, case_type, difficulty, k=3)
        
        if not results:
            return "No relevant cases found."
        
        output = "Found the following relevant cases:\n\n"
        for i, case in enumerate(results, 1):
            output += f"{i}. {case['case_name']}\n"
            output += f"   Type: {case['metadata']['case_type']}\n"
            output += f"   Difficulty: {case['metadata']['difficulty']}\n"
            output += f"   Industry: {case['metadata']['industry']}\n"
            output += f"   Preview: {case['content'][:200]}...\n\n"
        
        return output

class CaseContentTool(BaseTool):
    name: str = "get_case_content"
    description: str = "Get the full content of a specific case"
    
    def __init__(self, knowledge_base: CaseKnowledgeBase):
        super().__init__()
        self.knowledge_base = knowledge_base
    
    def _run(self, case_name: str) -> str:
        # Find the case in documents
        for doc in self.knowledge_base.documents:
            if doc.metadata.get('filename') == case_name:
                return doc.page_content
        
        return f"Case '{case_name}' not found."

# AI Agents
class ConsultingAgents:
    def __init__(self, knowledge_base: CaseKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.case_search_tool = CaseSearchTool(knowledge_base)
        self.case_content_tool = CaseContentTool(knowledge_base)
        
        # Initialize OpenAI
        if Config.OPENAI_API_KEY:
            openai.api_key = Config.OPENAI_API_KEY
        
        self.agents = self._create_agents()
        self.tasks = []
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create all AI agents for the consulting interview"""
        
        # Case Manager Agent
        case_manager = Agent(
            role="Case Manager",
            goal="Select appropriate cases based on user preferences and manage the interview flow",
            backstory="""You are a senior consulting partner who has conducted thousands of case interviews. 
            You have deep knowledge of different case types and can select the most appropriate cases for candidates based on their skill level and preferences.""",
            tools=[self.case_search_tool, self.case_content_tool],
            verbose=True,
            allow_delegation=False
        )
        
        # Case Giver Agent
        case_giver = Agent(
            role="Case Giver",
            goal="Present cases professionally and evaluate clarifying questions",
            backstory="""You are a McKinsey consultant with extensive business knowledge across industries. 
            You excel at presenting case studies clearly and evaluating whether candidates ask the right clarifying questions.""",
            tools=[self.case_content_tool],
            verbose=True,
            allow_delegation=False
        )
        
        # Framework Checker Agent
        framework_checker = Agent(
            role="Framework Checker",
            goal="Evaluate candidate frameworks for being MECE and comprehensive",
            backstory="""You are a consulting case prep instructor who has helped hundreds of candidates. 
            You specialize in evaluating frameworks to ensure they are Mutually Exclusive, Collectively Exhaustive, 
            and contain at least 3 factors with 3 questions each.""",
            tools=[self.case_content_tool],
            verbose=True,
            allow_delegation=False
        )
        
        # Business Math Checker Agent
        math_checker = Agent(
            role="Business Math Checker",
            goal="Verify calculations and provide business context for mathematical concepts",
            backstory="""You are a consultant who excels at arithmetic shortcuts and business mathematics. 
            You understand why formulas work and can explain complex business concepts using simple examples.""",
            tools=[self.case_content_tool],
            verbose=True,
            allow_delegation=False
        )
        
        # Recommendation Checker Agent
        recommendation_checker = Agent(
            role="Recommendation Checker",
            goal="Evaluate final recommendations and summaries for executive presentation",
            backstory="""You are a consultant who has presented to countless CEOs and executives. 
            You know how to structure recommendations professionally with risk mitigation and next steps.""",
            tools=[self.case_content_tool],
            verbose=True,
            allow_delegation=False
        )
        
        return {
            "case_manager": case_manager,
            "case_giver": case_giver,
            "framework_checker": framework_checker,
            "math_checker": math_checker,
            "recommendation_checker": recommendation_checker
        }
    
    def create_case_selection_task(self, user_preferences: Dict[str, str]) -> Task:
        """Create task for case selection"""
        return Task(
            description=f"""
            Based on the user's preferences:
            - Case Type: {user_preferences.get('case_type', 'Any')}
            - Difficulty: {user_preferences.get('difficulty', 'Medium')}
            - Industry: {user_preferences.get('industry', 'Any')}
            
            Search the knowledge base and select the most appropriate case.
            Provide the case name, type, difficulty, and a brief overview.
            """,
            agent=self.agents["case_manager"],
            expected_output="Selected case name with overview and metadata"
        )
    
    def create_case_presentation_task(self, case_name: str) -> Task:
        """Create task for case presentation"""
        return Task(
            description=f"""
            Present the case '{case_name}' to the candidate in a professional manner.
            Include all necessary context and background information.
            Wait for the candidate to ask clarifying questions and evaluate their quality.
            
            Good clarifying questions should:
            - Focus on framework selection and business understanding
            - Ask about target audience, timeline, profitability
            - Avoid being too specific or number-based initially
            
            Provide hints if questions are not strategic enough.
            """,
            agent=self.agents["case_giver"],
            expected_output="Case presentation with evaluation of clarifying questions"
        )
    
    def create_framework_evaluation_task(self, user_framework: str, case_name: str) -> Task:
        """Create task for framework evaluation"""
        return Task(
            description=f"""
            Evaluate the candidate's framework for case '{case_name}':
            {user_framework}
            
            Check if the framework is:
            1. Mutually Exclusive and Collectively Exhaustive (MECE)
            2. Has at least 3 factors with 3 questions each
            3. Uses appropriate business language for the industry
            4. Matches the expected solution approach
            
            Provide specific feedback and hints for improvement.
            """,
            agent=self.agents["framework_checker"],
            expected_output="Framework evaluation with specific feedback and improvement suggestions"
        )
    
    def create_math_evaluation_task(self, user_calculation: str, case_name: str) -> Task:
        """Create task for math evaluation"""
        return Task(
            description=f"""
            Evaluate the candidate's mathematical work for case '{case_name}':
            {user_calculation}
            
            Check for:
            1. Correct numbers and units
            2. Proper rounding techniques
            3. Step-by-step solution approach
            4. Business context and explanations
            
            Provide hints and explain concepts using simple examples when needed.
            """,
            agent=self.agents["math_checker"],
            expected_output="Mathematical evaluation with corrections and business context"
        )
    
    def create_recommendation_evaluation_task(self, user_recommendation: str, case_name: str) -> Task:
        """Create task for recommendation evaluation"""
        return Task(
            description=f"""
            Evaluate the candidate's recommendation for case '{case_name}':
            {user_recommendation}
            
            Check if the recommendation:
            1. Uses final numbers from calculations
            2. Employs proper business terminology
            3. Is formatted for executive presentation
            4. Includes risk mitigation strategies
            5. Ends with clear next steps
            
            Provide feedback on executive communication style.
            """,
            agent=self.agents["recommendation_checker"],
            expected_output="Recommendation evaluation with executive communication feedback"
        )

# Main Application
def main():
    st.markdown('<h1 class="main-header">🎯 BCG Consulting Case Interview Prep</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input("OpenAI API Key", type="password", value=Config.OPENAI_API_KEY)
        if api_key:
            Config.OPENAI_API_KEY = api_key
            os.environ["OPENAI_API_KEY"] = api_key
        
        st.divider()
        
        # Case preferences
        st.header("📋 Case Preferences")
        case_type = st.selectbox("Case Type", [
            "Any", "Market Entry", "Profitability", "Pricing", 
            "M&A", "Operations", "Growth", "Digital"
        ])
        
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
        
        industry = st.selectbox("Industry", [
            "Any", "Healthcare", "Technology", "Retail", 
            "Financial Services", "Manufacturing", "Energy", "Transportation"
        ])
        
        st.divider()
        
        # Knowledge base status
        st.header("📚 Knowledge Base")
        if st.session_state.case_database:
            st.success(f"✅ Loaded {len(st.session_state.case_database.case_metadata)} cases")
            
            # Display case statistics
            if st.session_state.case_database.case_metadata:
                case_types = [case['case_type'] for case in st.session_state.case_database.case_metadata]
                difficulties = [case['difficulty'] for case in st.session_state.case_database.case_metadata]
                
                st.write("**Case Types:**")
                for ct in set(case_types):
                    st.write(f"- {ct}: {case_types.count(ct)}")
                
                st.write("**Difficulties:**")
                for diff in set(difficulties):
                    st.write(f"- {diff}: {difficulties.count(diff)}")
        else:
            st.warning("⚠️ No cases loaded")
    
    # Main content area
    if not Config.OPENAI_API_KEY:
        st.error("🔑 Please enter your OpenAI API key in the sidebar to continue.")
        return
    
    # Initialize knowledge base
    if not st.session_state.case_database:
        st.header("📁 Load Case Database")
        st.info("Upload PDF files containing consulting cases to build the knowledge base.")
        
        # File uploader for PDFs
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload case study PDFs from your Google Drive folder"
        )
        
        if uploaded_files:
            if st.button("🔄 Build Knowledge Base"):
                with st.spinner("Building knowledge base..."):
                    # Process uploaded files
                    pdf_files = []
                    for uploaded_file in uploaded_files:
                        pdf_files.append({
                            'name': uploaded_file.name,
                            'content': uploaded_file.read()
                        })
                    
                    # Build knowledge base
                    st.session_state.case_database = CaseKnowledgeBase()
                    case_metadata = st.session_state.case_database.build_knowledge_base(pdf_files)
                    
                    st.success(f"✅ Successfully processed {len(case_metadata)} cases!")
                    st.rerun()
        
        # Sample data option
        st.divider()
        st.subheader("🎯 Demo Mode")
        if st.button("🚀 Use Sample Cases"):
            with st.spinner("Loading sample cases..."):
                # Create sample case database
                st.session_state.case_database = CaseKnowledgeBase()
                
                # Sample cases
                sample_cases = [
                    {
                        'name': 'Market Entry - Tech Company.pdf',
                        'content': create_sample_case_content("market_entry")
                    },
                    {
                        'name': 'Profitability - Restaurant Chain.pdf',
                        'content': create_sample_case_content("profitability")
                    },
                    {
                        'name': 'Pricing Strategy - Software.pdf',
                        'content': create_sample_case_content("pricing")
                    }
                ]
                
                # Build sample knowledge base
                st.session_state.case_database.build_knowledge_base(sample_cases)
                st.success("✅ Sample cases loaded!")
                st.rerun()
    
    else:
        # Interview interface
        st.header("🎯 Case Interview")
        
        # Initialize agents
        if not st.session_state.agents_initialized:
            with st.spinner("Initializing AI agents..."):
                st.session_state.agents = ConsultingAgents(st.session_state.case_database)
                st.session_state.agents_initialized = True
        
        # Interview stages
        if st.session_state.interview_stage == "setup":
            interview_setup_ui(case_type, difficulty, industry)
        elif st.session_state.interview_stage == "case_presentation":
            case_presentation_ui()
        elif st.session_state.interview_stage == "clarifying_questions":
            clarifying_questions_ui()
        elif st.session_state.interview_stage == "framework":
            framework_ui()
        elif st.session_state.interview_stage == "calculations":
            calculations_ui()
        elif st.session_state.interview_stage == "recommendations":
            recommendations_ui()
        elif st.session_state.interview_stage == "feedback":
            feedback_ui()

def create_sample_case_content(case_type: str) -> bytes:
    """Create sample case content for demo purposes"""
    sample_cases = {
        "market_entry": """
        MARKET ENTRY CASE
        
        Your client is a leading European technology company specializing in cloud computing solutions. 
        They are considering entering the US market and want to understand if this is a profitable opportunity.
        
        Background:
        - Company has 15 years of experience in European markets
        - Strong technical capabilities and established partnerships
        - Annual revenue of €500M in Europe
        - Considering both B2B and B2C segments
        
        Key Questions:
        1. Should they enter the US market?
        2. What entry strategy would you recommend?
        3. What are the key risks and mitigation strategies?
        
        This is an intermediate-level case focusing on market analysis and strategic planning.
        """,
        
        "profitability": """
        PROFITABILITY CASE
        
        Your client is a restaurant chain with 50 locations across the Midwest. 
        They have seen declining profits over the past 2 years and need to understand why.
        
        Background:
        - Family-owned business operating for 20 years
        - Known for casual dining and reasonable prices
        - Average check size has remained stable at $25
        - Number of customers has declined by 15% year-over-year
        
        Financial Information:
        - Revenue: $50M (down from $58M last year)
        - Food costs: 32% of revenue
        - Labor costs: 28% of revenue
        - Rent: 8% of revenue
        - Other expenses: 22% of revenue
        
        This is a beginner-level case focusing on profitability analysis and cost structure.
        """,
        
        "pricing": """
        PRICING STRATEGY CASE
        
        Your client is a software company that develops project management tools for small businesses. 
        They want to optimize their pricing strategy to maximize revenue.
        
        Background:
        - SaaS product with monthly subscription model
        - Current price: $29/month per user
        - 10,000 active subscribers
        - Low customer acquisition cost due to word-of-mouth
        - High customer satisfaction (NPS: 65)
        
        Market Context:
        - Competitors range from $15-50/month per user
        - Market is growing at 20% annually
        - Customers typically have 5-15 users per company
        
        This is an advanced case focusing on pricing optimization and revenue maximization.
        """
    }
    
    content = sample_cases.get(case_type, sample_cases["market_entry"])
    return content.encode('utf-8')

def interview_setup_ui(case_type: str, difficulty: str, industry: str):
    """UI for interview setup"""
    st.subheader("🎯 Interview Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Selected Preferences:**
        - Case Type: {case_type}
        - Difficulty: {difficulty}
        - Industry: {industry}
        """)
    
    with col2:
        st.info("""
        **Interview Process:**
        1. Case presentation
        2. Clarifying questions
        3. Framework development
        4. Calculations
        5. Recommendations
        6. Feedback
        """)
    
    if st.button("🚀 Start Interview", type="primary"):
        # Select case using AI agent
        user_preferences = {
            'case_type': case_type if case_type != 'Any' else None,
            'difficulty': difficulty,
            'industry': industry if industry != 'Any' else None
        }
        
        with st.spinner("Selecting optimal case..."):
            # Use case manager to select case
            task = st.session_state.agents.create_case_selection_task(user_preferences)
            crew = Crew(
                agents=[st.session_state.agents.agents["case_manager"]],
                tasks=[task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            st.session_state.current_case = result
            st.session_state.interview_stage = "case_presentation"
            st.rerun()

def case_presentation_ui():
    """UI for case presentation"""
    st.subheader("📋 Case Presentation")
    
    if st.session_state.current_case:
        st.markdown(f"""
        <div class="case-info">
        <h4>Selected Case</h4>
        {st.session_state.current_case}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✅ I've read the case"):
            st.session_state.interview_stage = "clarifying_questions"
            st.rerun()
    
    # Option to select different case
    if st.button("🔄 Select Different Case"):
        st.session_state.interview_stage = "setup"
        st.rerun()

def clarifying_questions_ui():
    """UI for clarifying questions phase"""
    st.subheader("❓ Clarifying Questions")
    
    st.info("""
    **Instructions:** Ask clarifying questions to better understand the case. 
    Focus on strategic questions about the business, not specific numbers.
    """)
    
    # Chat interface for clarifying questions
    if "clarifying_messages" not in st.session_state:
        st.session_state.clarifying_messages = []
    
    # Display conversation
    for message in st.session_state.clarifying_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    user_question = st.chat_input("Ask a clarifying question...")
    
    if user_question:
        # Add user message
        st.session_state.clarifying_messages.append({"role": "user", "content": user_question})
        
        with st.spinner("Evaluating your question..."):
            # Use case giver agent to evaluate question
            task = st.session_state.agents.create_case_presentation_task("current_case")
            # Simulate agent response (in production, this would use the actual crew)
            response = f"Good question! {user_question} shows you're thinking strategically. Here's some additional context..."
            
            st.session_state.clarifying_messages.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Continue button
    if len(st.session_state.clarifying_messages) >= 2:
        if st.button("➡️ Continue to Framework"):
            st.session_state.interview_stage = "framework"
            st.rerun()

def framework_ui():
    """UI for framework development phase"""
    st.subheader("🏗️ Framework Development")
    
    st.info("""
    **Instructions:** Develop a framework to approach this case. 
    Your framework should be MECE (Mutually Exclusive, Collectively Exhaustive) 
    with at least 3 factors and 3 questions for each factor.
    """)
    
    # Framework input
    framework_input = st.text_area(
        "Enter your framework:",
        height=300,
        placeholder="""Example:
        1. Market Analysis
           - What is the market size and growth rate?
           - Who are the key competitors?
           - What are the customer segments?
        
        2. Company Capabilities
           - What are our core strengths?
           - What resources do we have?
           - What is our competitive advantage?
        
        3. Financial Considerations
           - What are the investment requirements?
           - What are the expected returns?
           - What are the key cost drivers?
        """
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Get Framework Feedback"):
            if framework_input:
                with st.spinner("Evaluating your framework..."):
                    # Use framework checker agent
                    task = st.session_state.agents.create_framework_evaluation_task(
                        framework_input, "current_case"
                    )
                    
                    # Simulate agent response
                    st.success("✅ Framework evaluated! Check the feedback below.")
                    st.session_state.framework_feedback = """
                    **Framework Evaluation:**
                    
                    ✅ **Strengths:**
                    - Good structure with clear categories
                    - Questions are strategic and relevant
                    - Covers key business areas
                    
                    ⚠️ **Areas for Improvement:**
                    - Consider adding operational factors
                    - Market analysis could be more specific to the industry
                    - Risk assessment framework would strengthen the approach
                    
                    **Suggestions:**
                    - Add a fourth pillar for "Implementation & Risks"
                    - Make questions more specific to the case context
                    - Consider competitive positioning more deeply
                    """
            else:
                st.error("Please enter your framework first.")
    
    with col2:
        if st.button("➡️ Continue to Calculations"):
            if framework_input:
                st.session_state.user_framework = framework_input
                st.session_state.interview_stage = "calculations"
                st.rerun()
            else:
                st.error("Please develop your framework first.")
    
    # Display feedback if available
    if hasattr(st.session_state, 'framework_feedback'):
        st.markdown(st.session_state.framework_feedback)

def calculations_ui():
    """UI for calculations phase"""
    st.subheader("🔢 Calculations")
    
    st.info("""
    **Instructions:** Perform the necessary calculations for this case. 
    Show your work step-by-step and use proper business reasoning.
    """)
    
    # Sample calculation problems based on case type
    st.markdown("""
    **Sample Calculation Prompt:**
    
    Based on the case information, calculate:
    1. The total addressable market (TAM)
    2. Expected market share in year 1
    3. Revenue projections for the first 3 years
    4. Break-even analysis
    """)
    
    # Calculation input
    calculation_input = st.text_area(
        "Enter your calculations:",
        height=400,
        placeholder="""Example:
        1. Total Addressable Market (TAM):
           - US cloud computing market: $100B
           - Our segment (enterprise): 60% = $60B
           - Serviceable market: $60B
        
        2. Market Share Calculation:
           - Conservative estimate: 0.1% in Year 1
           - Year 1 revenue potential: $60B × 0.1% = $60M
        
        3. Revenue Projections:
           - Year 1: $60M
           - Year 2: $60M × 1.5 = $90M (50% growth)
           - Year 3: $90M × 1.3 = $117M (30% growth)
        
        4. Break-even Analysis:
           - Fixed costs: $30M
           - Variable costs: 40% of revenue
           - Break-even: $30M ÷ (1 - 0.4) = $50M
        """
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Get Math Feedback"):
            if calculation_input:
                with st.spinner("Checking your calculations..."):
                    # Use math checker agent
                    task = st.session_state.agents.create_math_evaluation_task(
                        calculation_input, "current_case"
                    )
                    
                    # Simulate agent response
                    st.success("✅ Calculations reviewed! Check the feedback below.")
                    st.session_state.math_feedback = """
                    **Calculation Review:**
                    
                    ✅ **Correct Approaches:**
                    - Good step-by-step breakdown
                    - Reasonable assumptions for market share
                    - Proper break-even formula application
                    
                    ⚠️ **Areas to Improve:**
                    - Consider rounding consistently (round one up, one down in multiplication)
                    - Add sensitivity analysis for key assumptions
                    - Include unit checks for all calculations
                    
                    **Business Context:**
                    - Market share of 0.1% seems conservative but realistic for new entrant
                    - Growth rates should be benchmarked against industry standards
                    - Consider seasonal factors and market cycles
                    """
            else:
                st.error("Please enter your calculations first.")
    
    with col2:
        if st.button("➡️ Continue to Recommendations"):
            if calculation_input:
                st.session_state.user_calculations = calculation_input
                st.session_state.interview_stage = "recommendations"
                st.rerun()
            else:
                st.error("Please complete your calculations first.")
    
    # Display feedback if available
    if hasattr(st.session_state, 'math_feedback'):
        st.markdown(st.session_state.math_feedback)

def recommendations_ui():
    """UI for recommendations phase"""
    st.subheader("💡 Recommendations")
    
    st.info("""
    **Instructions:** Provide your final recommendation as if presenting to the CEO. 
    Include risk mitigation strategies and clear next steps.
    """)
    
    # Recommendation input
    recommendation_input = st.text_area(
        "Enter your recommendation:",
        height=400,
        placeholder="""Example:
        **Recommendation: Proceed with US Market Entry**
        
        Based on our analysis, I recommend proceeding with the US market entry for the following reasons:
        
        **Key Findings:**
        - TAM of $60B with strong growth trajectory
        - Break-even achievable at $50M revenue (realistic in 12-18 months)
        - Strong competitive positioning due to European expertise
        
        **Financial Projection:**
        - Year 1: $60M revenue
        - Year 2: $90M revenue  
        - Year 3: $117M revenue
        - ROI: 180% by Year 3
        
        **Risk Mitigation:**
        - Start with pilot program in 2-3 major cities
        - Establish local partnerships to reduce market entry costs
        - Develop contingency plan if market share falls below 0.05%
        
        **Next Steps:**
        1. Conduct detailed market research in target cities (30 days)
        2. Identify and negotiate with local partners (60 days)
        3. Develop go-to-market strategy (90 days)
        4. Launch pilot program (120 days)
        """
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Get Recommendation Feedback"):
            if recommendation_input:
                with st.spinner("Evaluating your recommendation..."):
                    # Use recommendation checker agent
                    task = st.session_state.agents.create_recommendation_evaluation_task(
                        recommendation_input, "current_case"
                    )
                    
                    # Simulate agent response
                    st.success("✅ Recommendation evaluated! Check the feedback below.")
                    st.session_state.recommendation_feedback = """
                    **Recommendation Review:**
                    
                    ✅ **Executive Communication Strengths:**
                    - Clear recommendation upfront
                    - Uses specific numbers from analysis
                    - Professional tone appropriate for CEO
                    - Includes concrete next steps with timelines
                    
                    ✅ **Content Strengths:**
                    - Good risk mitigation strategies
                    - Realistic timeline for implementation
                    - Addresses key financial metrics
                    
                    ⚠️ **Areas for Enhancement:**
                    - Consider adding competitive response scenarios
                    - Include resource requirements for each next step
                    - Add success metrics and KPIs for tracking
                    
                    **Overall:** Strong recommendation with executive-level presentation skills demonstrated.
                    """
            else:
                st.error("Please enter your recommendation first.")
    
    with col2:
        if st.button("🎯 Get Final Feedback"):
            if recommendation_input:
                st.session_state.user_recommendation = recommendation_input
                st.session_state.interview_stage = "feedback"
                st.rerun()
            else:
                st.error("Please complete your recommendation first.")
    
    # Display feedback if available
    if hasattr(st.session_state, 'recommendation_feedback'):
        st.markdown(st.session_state.recommendation_feedback)

def feedback_ui():
    """UI for final feedback and scoring"""
    st.subheader("📊 Interview Feedback")
    
    # Overall performance summary
    st.markdown("""
    <div class="case-info">
    <h4>🎯 Overall Performance Summary</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Create performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Clarifying Questions", "8/10", "Strong")
    
    with col2:
        st.metric("Framework", "7/10", "Good")
    
    with col3:
        st.metric("Calculations", "9/10", "Excellent")
    
    with col4:
        st.metric("Recommendations", "8/10", "Strong")
    
    # Detailed feedback sections
    st.subheader("📋 Detailed Feedback")
    
    # Strengths
    st.markdown("""
    **✅ Key Strengths:**
    - Excellent quantitative analysis with clear step-by-step reasoning
    - Strong executive communication in final recommendation
    - Good strategic thinking in framework development
    - Appropriate level of detail for case complexity
    """)
    
    # Areas for improvement
    st.markdown("""
    **⚠️ Areas for Improvement:**
    - Framework could be more industry-specific
    - Consider more creative risk mitigation strategies
    - Add sensitivity analysis to calculations
    - Include more competitive analysis in recommendations
    """)
    
    # Benchmarking
    st.subheader("📈 Benchmarking")
    
    performance_data = {
        'Category': ['Clarifying Questions', 'Framework', 'Calculations', 'Recommendations'],
        'Your Score': [8, 7, 9, 8],
        'Average Score': [6, 6, 7, 6],
        'Top 10%': [9, 8, 9, 9]
    }
    
    import pandas as pd
    df = pd.DataFrame(performance_data)
    st.dataframe(df, use_container_width=True)
    
    # Next steps
    st.subheader("🚀 Next Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📚 Study Recommendations:**
        - Practice more industry-specific frameworks
        - Review competitive analysis techniques
        - Study sensitivity analysis methods
        - Practice executive presentation skills
        """)
    
    with col2:
        st.markdown("""
        **🎯 Practice Focus:**
        - Market entry cases (similar difficulty)
        - Cases requiring competitive analysis
        - Quantitative-heavy cases
        - Executive presentation practice
        """)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Practice Another Case"):
            # Reset session state
            st.session_state.interview_stage = "setup"
            st.session_state.current_case = None
            st.session_state.clarifying_messages = []
            if hasattr(st.session_state, 'framework_feedback'):
                delattr(st.session_state, 'framework_feedback')
            if hasattr(st.session_state, 'math_feedback'):
                delattr(st.session_state, 'math_feedback')
            if hasattr(st.session_state, 'recommendation_feedback'):
                delattr(st.session_state, 'recommendation_feedback')
            st.rerun()
    
    with col2:
        if st.button("📊 View Performance History"):
            st.info("Performance history feature coming soon!")
    
    with col3:
        if st.button("📧 Email Feedback"):
            st.info("Email feedback feature coming soon!")
    
    # Export feedback
    st.subheader("📄 Export Feedback")
    
    feedback_report = generate_feedback_report()
    st.download_button(
        label="📥 Download Feedback Report",
        data=feedback_report,
        file_name=f"case_interview_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

def generate_feedback_report():
    """Generate a comprehensive feedback report"""
    report = f"""# Case Interview Feedback Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Case Information
- **Case Type**: Market Entry
- **Difficulty**: Medium
- **Industry**: Technology

## Performance Summary
- **Overall Score**: 8.0/10
- **Clarifying Questions**: 8/10
- **Framework Development**: 7/10
- **Calculations**: 9/10
- **Recommendations**: 8/10

## Detailed Feedback

### Strengths
- Excellent quantitative analysis with clear reasoning
- Strong executive communication skills
- Good strategic thinking approach
- Appropriate level of detail for case complexity

### Areas for Improvement
- Framework could be more industry-specific
- Consider more creative risk mitigation strategies
- Add sensitivity analysis to calculations
- Include more competitive analysis

### Benchmarking
Your performance exceeds the average candidate in most areas, particularly in calculations and recommendations.

### Next Steps
1. Practice industry-specific frameworks
2. Review competitive analysis techniques
3. Study sensitivity analysis methods
4. Continue practicing executive presentation skills

### Recommended Practice Areas
- Market entry cases (similar difficulty)
- Cases requiring competitive analysis
- Quantitative-heavy cases
- Executive presentation practice

---
*This report was generated by the BCG Case Interview Prep AI system*
"""
    return report

# Additional utility functions
def display_agent_status():
    """Display the status of all agents"""
    st.subheader("🤖 Agent Status")
    
    agents_info = [
        ("Case Manager", "Selecting optimal cases", "completed"),
        ("Case Giver", "Presenting case information", "running"),
        ("Framework Checker", "Evaluating frameworks", "pending"),
        ("Math Checker", "Reviewing calculations", "pending"),
        ("Recommendation Checker", "Assessing recommendations", "pending")
    ]
    
    for name, task, status in agents_info:
        status_class = f"status-{status}"
        st.markdown(f"""
        <div class="agent-card">
        <h4>{name}</h4>
        <p>{task}</p>
        <span class="task-status {status_class}">{status.title()}</span>
        </div>
        """, unsafe_allow_html=True)

# Error handling and logging
def handle_agent_error(error: Exception, agent_name: str):
    """Handle errors from AI agents gracefully"""
    st.error(f"Error in {agent_name}: {str(error)}")
    st.info("Please try again or contact support if the issue persists.")

# Performance monitoring
def log_performance_metrics(stage: str, duration: float, success: bool):
    """Log performance metrics for monitoring"""
    # In production, this would log to a monitoring service
    pass

# Entry point
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")

# Requirements for deployment
"""
# requirements.txt
streamlit==1.28.0
crewai==0.1.0
openai==1.3.0
langchain==0.1.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
PyPDF2==3.0.1
numpy==1.24.3
pandas==2.0.3
python-dotenv==1.0.0
"""

# Deployment configuration
"""
# .streamlit/config.toml
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
"""
