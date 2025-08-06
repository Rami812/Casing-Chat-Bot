import os
import json
import io
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
from pdfminer.high_level import extract_text
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import TensorflowHubEmbeddings
from langchain.vectorstores import FAISS
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from typing import List, Dict, Any
import uuid
import tempfile

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-this')
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max file size

# Initialize Pinecone for cloud knowledge base
pc = None
index = None

def init_pinecone():
    """Initialize Pinecone vector database"""
    global pc, index
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print("Warning: PINECONE_API_KEY not set. Knowledge base will use local FAISS.")
            return None
            
        pc = Pinecone(api_key=api_key)
        index_name = "case-interview-kb"
        
        # Create index if it doesn't exist
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=512,  # TensorFlow Hub embeddings dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        index = pc.Index(index_name)
        print(f"Connected to Pinecone index: {index_name}")
        return index
    except Exception as e:
        print(f"Failed to initialize Pinecone: {e}")
        return None

# Initialize the LLM for CrewAI agents
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    my_llm = LLM(
        api_key=api_key,
        model="gemini/gemini-1.5-flash"
    )
    return my_llm

# Custom tool for knowledge base search
@tool
def search_case_knowledge(query: str) -> str:
    """Search the uploaded case document for relevant information."""
    session_id = session.get('session_id')
    if not session_id:
        return "No active session found"
    
    # Try Pinecone first, fallback to local storage
    if index:
        try:
            # Use embeddings to search Pinecone
            embeddings = TensorflowHubEmbeddings()
            query_embedding = embeddings.embed_query(query)
            
            results = index.query(
                vector=query_embedding,
                filter={"session_id": session_id},
                top_k=3,
                include_metadata=True
            )
            
            if results['matches']:
                return "\n".join([match['metadata']['text'] for match in results['matches']])
        except Exception as e:
            print(f"Pinecone search error: {e}")
    
    # Fallback to session-based local storage
    knowledge_base = session.get('knowledge_base')
    if knowledge_base:
        # For local storage, we'll implement a simple text search
        documents = knowledge_base.get('documents', [])
        relevant_docs = [doc for doc in documents if query.lower() in doc.lower()][:3]
        return "\n".join(relevant_docs)
    
    return "No knowledge base available"

# Custom tool for case stage tracking
@tool
def get_current_stage() -> str:
    """Get the current stage of the case interview."""
    return session.get('current_stage', 'case_introduction')

@tool
def update_stage(new_stage: str) -> str:
    """Update the current stage of the case interview."""
    session['current_stage'] = new_stage
    return f"Stage updated to: {new_stage}"

class CaseInterviewCrew:
    def __init__(self):
        self.llm = get_llm()
        self.agents = self._create_agents()
        self.tasks = self._create_tasks()
        
    def _create_agents(self):
        # Case Giver Agent
        case_giver = Agent(
            role="Case Giver",
            goal="Present case scenarios and validate clarifying questions from students",
            backstory="""You are a McKinsey consultant with extensive business knowledge and industry expertise. 
            You specialize in presenting case studies and evaluating the quality of clarifying questions. 
            You ensure students ask strategic, framework-oriented questions rather than diving into specific numbers.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[search_case_knowledge, get_current_stage, update_stage]
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
            llm=self.llm,
            tools=[search_case_knowledge, get_current_stage]
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
            llm=self.llm,
            tools=[search_case_knowledge, get_current_stage]
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
            llm=self.llm,
            tools=[search_case_knowledge, get_current_stage]
        )
        
        return {
            'case_giver': case_giver,
            'framework_checker': framework_checker,
            'math_checker': math_checker,
            'recommendation_checker': recommendation_checker
        }
    
    def _create_tasks(self):
        return {
            'case_introduction': self._create_case_introduction_task(),
            'clarifying_questions': self._create_clarifying_questions_task(),
            'framework_evaluation': self._create_framework_evaluation_task(),
            'math_evaluation': self._create_math_evaluation_task(),
            'recommendation_evaluation': self._create_recommendation_evaluation_task()
        }
    
    def _create_case_introduction_task(self):
        return Task(
            description="""Present the case scenario from the uploaded document to the student. 
            Provide a clear, structured case prompt that includes:
            1. The business situation
            2. The key challenge or question
            3. Any initial context needed
            
            Keep the presentation engaging and professional, as if you're conducting a real McKinsey interview.""",
            agent=self.agents['case_giver'],
            expected_output="A well-structured case presentation with clear problem statement"
        )
    
    def _create_clarifying_questions_task(self):
        return Task(
            description="""Evaluate the student's clarifying questions based on these criteria:
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
            
            Respond with feedback and either answer good questions or redirect poor ones.""",
            agent=self.agents['case_giver'],
            expected_output="Feedback on clarifying questions with answers or redirection"
        )
    
    def _create_framework_evaluation_task(self):
        return Task(
            description="""Evaluate the student's framework against these criteria:
            1. Is it MECE (Mutually Exclusive, Collectively Exhaustive)?
            2. Does it have at least 3 main factors?
            3. Are there 3+ questions for each factor?
            4. Does it match the solution approach reasonably?
            5. Does it use appropriate business terminology for the industry?
            
            Example good frameworks:
            - Profitability framework (Revenue, Costs, Market factors)
            - Market Entry framework (Market, Competition, Company capabilities)
            
            Provide specific feedback on what's missing or could be improved. 
            Use industry-specific language (e.g., "number of seats" for theaters, not just "quantity").""",
            agent=self.agents['framework_checker'],
            expected_output="Detailed framework evaluation with specific improvement suggestions"
        )
    
    def _create_math_evaluation_task(self):
        return Task(
            description="""Check the student's mathematical work for:
            1. Correct numbers and units usage
            2. Proper rounding techniques (round one up, one down in multiplication; both down in division)
            3. Step-by-step solution approach
            4. Logical flow of calculations
            
            If the student asks for explanations, use simple analogies:
            - Cost of Goods Sold → pizza ingredients cost
            - Economies of scale → bulk buying discounts
            - Market share → slice of the total pizza
            
            Provide hints for errors and detailed explanations when requested.""",
            agent=self.agents['math_checker'],
            expected_output="Mathematical verification with explanations and corrections"
        )
    
    def _create_recommendation_evaluation_task(self):
        return Task(
            description="""Evaluate the student's final recommendation for:
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
            
            Provide feedback on tone, structure, and completeness.""",
            agent=self.agents['recommendation_checker'],
            expected_output="Comprehensive evaluation of recommendation with improvement suggestions"
        )
    
    def run_appropriate_task(self, user_input: str, chat_history: List) -> str:
        """Determine which task to run based on current stage and user input."""
        current_stage = session.get('current_stage', 'case_introduction')
        
        # Determine the appropriate task based on stage and input analysis
        if current_stage == 'case_introduction':
            if 'framework' in user_input.lower() or 'approach' in user_input.lower():
                session['current_stage'] = 'framework_evaluation'
                task = self._create_framework_evaluation_task()
            elif any(question_word in user_input.lower() for question_word in ['who', 'what', 'where', 'when', 'why', 'how']):
                session['current_stage'] = 'clarifying_questions'
                task = self._create_clarifying_questions_task()
            else:
                task = self._create_case_introduction_task()
        elif current_stage == 'clarifying_questions':
            if 'framework' in user_input.lower() or 'approach' in user_input.lower():
                session['current_stage'] = 'framework_evaluation'
                task = self._create_framework_evaluation_task()
            else:
                task = self._create_clarifying_questions_task()
        elif current_stage == 'framework_evaluation':
            if any(calc_word in user_input.lower() for calc_word in ['calculate', 'math', 'number', '$', '%']):
                session['current_stage'] = 'math_evaluation'
                task = self._create_math_evaluation_task()
            else:
                task = self._create_framework_evaluation_task()
        elif current_stage == 'math_evaluation':
            if any(rec_word in user_input.lower() for rec_word in ['recommend', 'suggestion', 'conclusion', 'summary']):
                session['current_stage'] = 'recommendation_evaluation'
                task = self._create_recommendation_evaluation_task()
            else:
                task = self._create_math_evaluation_task()
        else:  # recommendation_evaluation
            task = self._create_recommendation_evaluation_task()
        
        # Create a crew with the appropriate task
        crew = Crew(
            agents=[task.agent],
            tasks=[task],
            verbose=True,
            process=Process.sequential
        )
        
        # Add user input context to the task
        task.description += f"\n\nUser Input: {user_input}\nChat History: {str(chat_history[-5:])}"
        
        # Execute the task
        result = crew.kickoff()
        return result

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/start_session', methods=['POST'])
def start_session():
    """Initialize a new interview session"""
    session['session_id'] = str(uuid.uuid4())
    session['current_stage'] = 'case_introduction'
    session['chat_history'] = []
    session['knowledge_base'] = None
    
    return jsonify({
        'status': 'success',
        'message': 'Session started successfully',
        'session_id': session['session_id']
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF file upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        try:
            # Extract text from PDF
            pdf_bytes = file.read()
            text = extract_text(io.BytesIO(pdf_bytes))
            
            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            
            # Store in cloud or local knowledge base
            session_id = session.get('session_id')
            if index and session_id:
                # Store in Pinecone
                try:
                    embeddings = TensorflowHubEmbeddings()
                    vectors = []
                    
                    for i, chunk in enumerate(chunks):
                        vector_id = f"{session_id}_{i}"
                        embedding = embeddings.embed_query(chunk)
                        vectors.append({
                            'id': vector_id,
                            'values': embedding,
                            'metadata': {
                                'text': chunk,
                                'session_id': session_id,
                                'chunk_index': i
                            }
                        })
                    
                    # Upsert vectors in batches
                    batch_size = 100
                    for i in range(0, len(vectors), batch_size):
                        batch = vectors[i:i + batch_size]
                        index.upsert(vectors=batch)
                    
                    print(f"Stored {len(chunks)} chunks in Pinecone for session {session_id}")
                    
                except Exception as e:
                    print(f"Error storing in Pinecone: {e}")
                    # Fallback to local storage
                    session['knowledge_base'] = {'documents': chunks}
            else:
                # Store locally in session
                session['knowledge_base'] = {'documents': chunks}
            
            return jsonify({
                'status': 'success',
                'message': f'PDF processed successfully. {len(chunks)} chunks created.',
                'filename': secure_filename(file.filename)
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not session.get('session_id'):
        return jsonify({'error': 'No active session. Please start a new session.'}), 400
    
    if not session.get('knowledge_base') and not index:
        return jsonify({'error': 'Please upload a case study PDF first.'}), 400
    
    try:
        # Initialize crew
        crew = CaseInterviewCrew()
        
        # Get chat history
        chat_history = session.get('chat_history', [])
        
        # Process with CrewAI
        response = crew.run_appropriate_task(user_message, chat_history)
        
        # Update chat history
        chat_history.append({'user': user_message, 'assistant': response.raw})
        session['chat_history'] = chat_history[-10:]  # Keep last 10 exchanges
        
        # Get current stage info
        current_stage = session.get('current_stage', 'case_introduction')
        stage_names = {
            'case_introduction': 'Stage 1 - Case Introduction',
            'clarifying_questions': 'Stage 2 - Clarifying Questions',
            'framework_evaluation': 'Stage 3 - Framework Development',
            'math_evaluation': 'Stage 4 - Quantitative Analysis',
            'recommendation_evaluation': 'Stage 5 - Recommendations'
        }
        
        return jsonify({
            'response': response.raw,
            'current_stage': current_stage,
            'stage_display': stage_names.get(current_stage, 'Unknown Stage')
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing message: {str(e)}'}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset the current session"""
    session_id = session.get('session_id')
    
    # Clean up Pinecone vectors if they exist
    if index and session_id:
        try:
            # Delete vectors for this session
            index.delete(filter={"session_id": session_id})
        except Exception as e:
            print(f"Error cleaning up Pinecone vectors: {e}")
    
    # Clear session
    session.clear()
    
    return jsonify({'status': 'success', 'message': 'Session reset successfully'})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'pinecone_available': index is not None,
        'session_active': bool(session.get('session_id'))
    })

if __name__ == '__main__':
    # Initialize Pinecone
    init_pinecone()
    
    # Run Flask app
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(host=host, port=port, debug=debug)