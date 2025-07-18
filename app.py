import os
import chainlit as cl
from pdfminer.high_level import extract_text
import io
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import TensorflowHubEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from typing import List, Dict, Any
import json
import asyncio

# Initialize the LLM for CrewAI agents
def get_llm():
    # Use environment variable instead of hardcoded API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    my_llm = LLM(
        api_key=api_key,
        model="gemini/gemini-1.5-flash"
    )
    return my_llm

# Global knowledge base storage
knowledge_base = None

# Custom tool for knowledge base search
@tool
def search_case_knowledge(query: str) -> str:
    """Search the uploaded case document for relevant information."""
    if knowledge_base is not None:
        docs = knowledge_base.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in docs])
    return "No knowledge base available"

# Custom tool for case stage tracking
@tool
def get_current_stage() -> str:
    """Get the current stage of the case interview."""
    user_session = cl.user_session.get("user_session", {})
    return user_session.get('current_stage', 'case_introduction')

@tool
def update_stage(new_stage: str) -> str:
    """Update the current stage of the case interview."""
    user_session = cl.user_session.get("user_session", {})
    user_session['current_stage'] = new_stage
    cl.user_session.set("user_session", user_session)
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
        user_session = cl.user_session.get("user_session", {})
        current_stage = user_session.get('current_stage', 'case_introduction')
        
        # Determine the appropriate task based on stage and input analysis
        if current_stage == 'case_introduction':
            if 'framework' in user_input.lower() or 'approach' in user_input.lower():
                user_session['current_stage'] = 'framework_evaluation'
                cl.user_session.set("user_session", user_session)
                task = self._create_framework_evaluation_task()
            elif any(question_word in user_input.lower() for question_word in ['who', 'what', 'where', 'when', 'why', 'how']):
                user_session['current_stage'] = 'clarifying_questions'
                cl.user_session.set("user_session", user_session)
                task = self._create_clarifying_questions_task()
            else:
                task = self._create_case_introduction_task()
        elif current_stage == 'clarifying_questions':
            if 'framework' in user_input.lower() or 'approach' in user_input.lower():
                user_session['current_stage'] = 'framework_evaluation'
                cl.user_session.set("user_session", user_session)
                task = self._create_framework_evaluation_task()
            else:
                task = self._create_clarifying_questions_task()
        elif current_stage == 'framework_evaluation':
            if any(calc_word in user_input.lower() for calc_word in ['calculate', 'math', 'number', '$', '%']):
                user_session['current_stage'] = 'math_evaluation'
                cl.user_session.set("user_session", user_session)
                task = self._create_math_evaluation_task()
            else:
                task = self._create_framework_evaluation_task()
        elif current_stage == 'math_evaluation':
            if any(rec_word in user_input.lower() for rec_word in ['recommend', 'suggestion', 'conclusion', 'summary']):
                user_session['current_stage'] = 'recommendation_evaluation'
                cl.user_session.set("user_session", user_session)
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

# Initialize global crew
crew = None

@cl.on_chat_start
async def start():
    """Initialize the chat session when a user connects."""
    global crew, knowledge_base
    
    # Initialize user session
    cl.user_session.set("user_session", {"current_stage": "case_introduction"})
    
    # Send welcome message with file upload request
    await cl.Message(
        content="""# 🎯 Welcome to AI-Powered Case Interview Practice!

I'm your AI case interview coach powered by a team of expert consultants. 

To begin your practice session, please upload a case study PDF using the attachment button below. 

Once uploaded, I'll process the document and we can start your interview!

## 📋 Interview Stages Guide:
- **Stage 1**: Case Introduction - Listen and ask clarifying questions
- **Stage 2**: Framework Development - Create a MECE framework  
- **Stage 3**: Quantitative Analysis - Perform calculations
- **Stage 4**: Recommendations - Provide executive summary

## 💡 Tips for Success:
- **Clarifying Questions**: Focus on business context, not specific numbers
- **Framework**: Use MECE principles (Mutually Exclusive, Collectively Exhaustive)
- **Math**: Show your work step-by-step with proper rounding
- **Recommendations**: Speak as if presenting to a CEO

Upload your case study when you're ready to begin! 📄"""
    ).send()
    
    # Wait for file upload
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload your case study PDF to begin the interview session:",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=300
        ).send()
    
    if files:
        await process_uploaded_file(files[0])

async def process_uploaded_file(file):
    """Process the uploaded PDF file and initialize the knowledge base."""
    global crew, knowledge_base
    
    # Show processing message
    processing_msg = cl.Message(content="🔄 Processing your case study PDF...")
    await processing_msg.send()
    
    try:
        # Extract text from PDF
        pdf_bytes = file.content
        text = extract_text(io.BytesIO(pdf_bytes))
        
        # Update processing message
        processing_msg.content = "📝 Extracting text and creating knowledge base..."
        await processing_msg.update()
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings and knowledge base
        embeddings = TensorflowHubEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # Initialize CrewAI system
        crew = CaseInterviewCrew()
        
        # Update processing message to success
        processing_msg.content = f"✅ **Case study '{file.name}' processed successfully!**\n\nYour interview session is now ready. Let's begin with the case introduction!"
        await processing_msg.update()
        
        # Get current stage info
        user_session = cl.user_session.get("user_session", {})
        current_stage = user_session.get('current_stage', 'case_introduction')
        
        # Send stage indicator
        await cl.Message(
            content=f"🎯 **Current Stage**: Stage 1 - Case Introduction\n\nI'm now ready to present your case study. Type 'start' or 'begin' to receive the case prompt, or ask me any questions about the process."
        ).send()
        
    except Exception as e:
        # Handle processing errors
        processing_msg.content = f"❌ **Error processing PDF**: {str(e)}\n\nPlease try uploading the file again or contact support if the issue persists."
        await processing_msg.update()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages from users."""
    global crew, knowledge_base
    
    if crew is None or knowledge_base is None:
        await cl.Message(
            content="⚠️ **Please upload a case study PDF first!**\n\nUse the attachment button to upload your PDF file before we can begin the interview."
        ).send()
        return
    
    # Show typing indicator
    async with cl.Step(name="Consulting with expert agents", type="run") as step:
        try:
            # Get chat history
            chat_history = cl.chat_context.to_openai()
            
            # Process with CrewAI
            response = crew.run_appropriate_task(message.content, chat_history)
            
            # Get current stage for display
            user_session = cl.user_session.get("user_session", {})
            current_stage = user_session.get('current_stage', 'case_introduction')
            
            stage_names = {
                'case_introduction': 'Stage 1 - Case Introduction',
                'clarifying_questions': 'Stage 2 - Clarifying Questions',
                'framework_evaluation': 'Stage 3 - Framework Development',
                'math_evaluation': 'Stage 4 - Quantitative Analysis',
                'recommendation_evaluation': 'Stage 5 - Recommendations'
            }
            
            stage_display = stage_names.get(current_stage, 'Unknown Stage')
            
            # Format response with stage information
            formatted_response = f"🎯 **{stage_display}**\n\n{response.raw}"
            
            step.output = formatted_response
            
        except Exception as e:
            error_msg = f"❌ **Error processing your request**: {str(e)}\n\nPlease try rephrasing your question or contact support if the issue persists."
            step.output = error_msg

@cl.on_stop
async def on_stop():
    """Clean up when the chat session ends."""
    global crew, knowledge_base
    crew = None
    knowledge_base = None
    print("Chat session ended and resources cleaned up.")

# Configuration for Chainlit
@cl.cache
def get_app_config():
    return {
        "name": "AI Case Interview Coach",
        "description": "Multi-agent system for comprehensive case interview preparation",
        "author": "AI Case Interview Team",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Use production settings
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    # Run the Chainlit app with production settings
    cl.run(debug=False, host=host, port=port)
