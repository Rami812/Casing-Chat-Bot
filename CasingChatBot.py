import streamlit as st
from PyPDF import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import os
from langchain_community.document_loaders import JSONLoader
from langchain.embeddings import TensorflowHubEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
history = StreamlitChatMessageHistory(key="chat_messages")
#Look into how to make it look better with streamlit


def main():

    st.set_page_config(page_title="Give me a case to interview you")
    st.header("Ask your PDF")

    pdf=st.file_uploader("Upload your PDF",type="pdf")
    #extract the text
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        #st.write(text)
        #split the pdf text into chunks
        text_splitter= CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len

        )
        chunks=text_splitter.split_text(text)

        #Create embeddings
        

        embeddings = TensorflowHubEmbeddings()
        #embeddings=LlamaCppEmbeddings()
        knowledge_base=FAISS.from_texts(chunks, embeddings)
       

                #New Methods

                #get response 
        def get_response(query,chat_history,text,relevant_info):
                     additional_instructions="Do not provide the student/user the right answer unless they ask for it. Intesead ask the student to come up with the relevant part of the solution and provide it to check. Give the user the relevant numbers they need but wait for them to give you a solution."
                     template="You are doing a mock casing interview for college students that are beginners. The text in --are cases that you have done yourself."+"--"+text+"--"+additional_instructions+"After each user input, are to tell the user if they provided a correct answer or not. You will try to get to the correct solution yourself.Then check the relevant part for the solution which is between dashes. The text between dashes is the solution to solving the student's question--"+relevant_info+"--If a student says they are struggling help a student. Chat history:{chat_history} User question:{user_question}"

                     prompt=ChatPromptTemplate.from_template(template)
                     llm=ChatGroq(model="llama3-8b-8192",temperature=0,max_tokens=None, timeout=None,max_retries=2,api_key="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")
                     chain=prompt | llm | StrOutputParser()
                     return chain.invoke({"chat_history":chat_history, "user_question":query})
        #api_key=os.environ.get('GROQ_API_KEY')
        
        def find_relevant_text(query):
               query_str=query|StrOutputParser
               docs=knowledge_base.similarity_search(query_str)
               
        #Default sesssion state
        if "chat_history" not in st.session_state:
               st.session_state.chat_history=[AIMessage(content="Hello, I am a chatbot that helps students prepare for consulting casing interviews. How can I help you?")]
               

                      
                

                #Conversation
        for message in st.session_state.chat_history:
                     if isinstance(message,HumanMessage):
                          with st.chat_message("Human"):
                               st.markdown(message.content)
                     else:
                          with st.chat_message("AI"):
                               st.markdown(message.content)

                #User input
        user_query=st.chat_input("Feel free to ask questions or give an answer. ")
        if user_query is not None and user_query !="":
                     st.session_state.chat_history.append(HumanMessage(user_query))

                     with st.chat_message("Human"):
                          st.markdown(user_query)
                     with st.chat_message("AI"):
                          docs=knowledge_base.similarity_search(user_query)
                          ai_response=get_response(user_query,st.session_state.chat_history,text,str(docs[0])+str(docs[1])+str(docs[2]))

                          st.markdown(ai_response)
                     st.session_state.chat_history.append(AIMessage(ai_response))


                






if __name__=='__main__':
    main()
