__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from pdfminer.high_level import extract_text
import io
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import TensorflowHubEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import os
from crewai import Agent, Task, Crew, Process,LLM
from crewai.tools import tool
from typing import List, Dict, Any
import json

#from langchain_google_genai import GoogleGenerativeAI as genai
#import google.generativeai as genai
#from langchain_groq import ChatGroq

# Initialize the LLM for CrewAI agents
def get_llm():
  my_llm=LLM(api_key="AIzaSyDmLDKKNDS8J6J_lCIKG7VjXvMCw4vpUgs",
  model="gemini/gemini-1.5-flash")
  return my_llm


# Custom tool for knowledge base search
@tool
def search_case_knowledge(query: str) -> str:
    """Search the uploaded case document for relevant information."""
    if 'knowledge_base' in st.session_state:
        docs = st.session_state.knowledge_base.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in docs])
    return "No knowledge base available"

# Custom tool for case stage tracking
@tool
def get_current_stage() -> str:
    """Get the current stage of the case interview."""
    return st.session_state.get('current_stage', 'case_introduction')

@tool
def update_stage(new_stage: str) -> str:
    """Update the current stage of the case interview."""
    st.session_state['current_stage'] = new_stage
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
        current_stage = st.session_state.get('current_stage', 'case_introduction')
        
        # Determine the appropriate task based on stage and input analysis
        if current_stage == 'case_introduction':
            if 'framework' in user_input.lower() or 'approach' in user_input.lower():
                st.session_state['current_stage'] = 'framework_evaluation'
                task = self._create_framework_evaluation_task()
            elif any(question_word in user_input.lower() for question_word in ['who', 'what', 'where', 'when', 'why', 'how']):
                st.session_state['current_stage'] = 'clarifying_questions'
                task = self._create_clarifying_questions_task()
            else:
                task = self._create_case_introduction_task()
        elif current_stage == 'clarifying_questions':
            if 'framework' in user_input.lower() or 'approach' in user_input.lower():
                st.session_state['current_stage'] = 'framework_evaluation'
                task = self._create_framework_evaluation_task()
            else:
                task = self._create_clarifying_questions_task()
        elif current_stage == 'framework_evaluation':
            if any(calc_word in user_input.lower() for calc_word in ['calculate', 'math', 'number', '$', '%']):
                st.session_state['current_stage'] = 'math_evaluation'
                task = self._create_math_evaluation_task()
            else:
                task = self._create_framework_evaluation_task()
        elif current_stage == 'math_evaluation':
            if any(rec_word in user_input.lower() for rec_word in ['recommend', 'suggestion', 'conclusion', 'summary']):
                st.session_state['current_stage'] = 'recommendation_evaluation'
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

def main():
    st.set_page_config(page_title="CrewAI Case Interview Preparation", layout="wide")
    st.header("🎯 AI-Powered Case Interview Practice")
    st.subheader("Multi-Agent System for Comprehensive Case Preparation")
    
    # Initialize session state
    if 'crew' not in st.session_state:
        st.session_state.crew = None
    if 'current_stage' not in st.session_state:
        st.session_state.current_stage = 'case_introduction'
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm your AI case interview coach powered by a team of expert consultants. Please upload a case study PDF to begin your practice session.")
        ]
    
    # Sidebar for case upload and stage tracking
    with st.sidebar:
        st.header("📄 Case Upload")
        pdf = st.file_uploader("Upload your case study PDF", type="pdf")
        
        if pdf is not None:
            with st.spinner("Processing case document..."):
                # Extract text from PDF
                pdf_bytes = pdf.read()
                text = extract_text(io.BytesIO(pdf_bytes))
                
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
                st.session_state.knowledge_base = FAISS.from_texts(chunks, embeddings)
                
                # Initialize CrewAI system
                st.session_state.crew = CaseInterviewCrew()
                
                st.success("✅ Case document processed successfully!")
                st.info("You can now start your case interview practice.")
        
        # Display current stage
        st.header("🎯 Current Stage")
        stage_display = {
            'case_introduction': '1. Case Introduction',
            'clarifying_questions': '2. Clarifying Questions',
            'framework_evaluation': '3. Framework Development',
            'math_evaluation': '4. Quantitative Analysis',
            'recommendation_evaluation': '5. Recommendations'
        }
        current_stage = st.session_state.get('current_stage', 'case_introduction')
        st.write(f"**{stage_display.get(current_stage, 'Unknown')}**")
        
        # Stage descriptions
        st.header("📋 Interview Stages")
        with st.expander("Stage Guide"):
            st.markdown("""
            **1. Case Introduction**: Listen to the case and ask clarifying questions
            **2. Clarifying Questions**: Ask strategic, framework-oriented questions
            **3. Framework Development**: Create a MECE framework with 3+ factors
            **4. Quantitative Analysis**: Perform calculations with proper technique
            **5. Recommendations**: Provide executive summary with next steps
            """)
    
    # Main chat interface
    st.header("💬 Case Interview Session")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)
    
    # User input
    user_input = st.chat_input("Type your response, questions, or request guidance...")
    
    if user_input and st.session_state.crew:
        # Add user message to history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process with CrewAI
        with st.chat_message("assistant"):
            with st.spinner("Consulting with expert agents..."):
                try:
                    response = st.session_state.crew.run_appropriate_task(
                        user_input, 
                        st.session_state.chat_history
                    )
                    st.markdown(response)
                    
                    # Add AI response to history
                    st.session_state.chat_history.append(AIMessage(content=response.raw))
                    
                except Exception as e:
                    error_msg = f"I encountered an error while processing your request: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(AIMessage(content=error_msg))
    
    elif user_input and not st.session_state.crew:
        st.warning("Please upload a case study PDF first to begin the interview.")
    
    # Footer with tips
    with st.expander("💡 Tips for Success"):
        st.markdown("""
        **Clarifying Questions**: Focus on understanding the business, not specific numbers
        **Framework**: Use MECE principles - Mutually Exclusive, Collectively Exhaustive
        **Math**: Show your work step-by-step and use proper rounding techniques
        **Recommendations**: Speak as if presenting to a CEO - professional and actionable
        """)

if __name__ == '__main__':
    main()
