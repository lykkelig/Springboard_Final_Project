'''
What is Generative AI:
Generative AI refers to a class of artificial intelligence (AI) algorithms and models that 
    have the capability to generate new content or data resembling original input. 
    These models are trained on large datasets and learn patterns and structures within the 
    data, enabling them to generate new content that is often indistinguishable from 
    human-created content. Generative AI encompasses various techniques, including generative 
    adversarial networks (GANs), autoregressive models, and transformers.
    
What is langchain:
Langchain is a software framework designed for natural language processing (NLP) tasks and 
    applications. It provides a modular and extensible architecture that allows developers to 
    build and deploy various NLP models, algorithms, and pipelines. Langchain typically includes 
    components such as message parsers, language models, output generators, and document loaders, 
    among others. It aims to streamline the development and deployment of NLP solutions by providing 
    a standardized interface for interacting with different components and integrating them into 
    cohesive systems. Langchain can be used for a wide range of NLP applications, including text 
    classification, information retrieval, question answering, chatbots, and more.      
'''
# Import pandas library for data manipulation.
import pandas as pd
# Import function to load environment variables from a .env file.
from dotenv import load_dotenv
# Import classes for AIMessage and HumanMessage from langchain_core module.
from langchain_core.messages import AIMessage, HumanMessage
# Import class for RunnablePassthrough from langchain_core module.
from langchain_core.runnables import RunnablePassthrough
# Import class for StrOutputParser from langchain_core module.
from langchain_core.output_parsers import StrOutputParser
# Import OpenAI class from langchain_openai module.
from langchain_openai import OpenAI
# Import streamlit library for building interactive web applications.
import streamlit as st
# Import OpenAI class from langchain_community.llms module.
from langchain_community.llms import OpenAI
# Import FAISS class from langchain_community.vectorstores module.
from langchain_community.vectorstores import FAISS
# Import function to load a question-answering chain from langchain module.
from langchain.chains.question_answering import load_qa_chain
# Import DirectoryLoader class from langchain_community.document_loaders module.
from langchain_community.document_loaders import DirectoryLoader
# Import Python ODBC driver
import pyodbc
# Import function to create a SQL agent from langchain.agents module.
from langchain.agents import create_sql_agent
# Import function to create a SQL agent from langchain_community.agent_toolkits module.
from langchain_community.agent_toolkits import create_sql_agent
# Import SQLDatabaseToolkit class from langchain.agents.agent_toolkits module.
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# Import AgentType class from langchain.agents.agent_types module.
from langchain.agents.agent_types import AgentType
# Import ChatOpenAI class from langchain.chat_models module.
from langchain.chat_models import ChatOpenAI
# Import SQLDatabase class from langchain.sql_database module.
from langchain.sql_database import SQLDatabase
# Import ChatPromptTemplate class from langchain.prompts.chat module.
from langchain.prompts.chat import ChatPromptTemplate
# Import create_engine function from sqlalchemy module.
from sqlalchemy import create_engine

# Import secret API keys for OPENAI
load_dotenv()

# Database Connection to local SQL Server DB
def connect_to_database():
    conn = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-AVM38KQ\MSSQLSERVER01;DATABASE=MED_DATA')
    return conn

# Database Query Logic for a Specific patient using PATIENT_ID
def handle_database_query(question, patient_id):
    try:
        conn = connect_to_database()
        cursor = conn.cursor()

        # Static SQL to get patient data by ID (Needs Improvement)
        sql_query = """
            SELECT PT.PATIENT_ID, PT.NAME, PT.DATE_OF_BIRTH, PN.DESCRIPTION, 
            PN.TRANSCRIPTION, PN.KEYWORDS 
            FROM PATIENT AS PT 
            JOIN PATIENT_NOTES AS PN ON PT.PATIENT_ID = PN.PATIENT_ID
            WHERE PT.PATIENT_ID = ? 
            """
        cursor.execute(sql_query, (patient_id,))  # Pass patient_id as a parameter
        result = cursor.fetchall()
        # print(type(result))  # What type does this print?
        # print(type(result[0])) # And the type of the first element within result? 
        formatted_result = process_database_result(result)  # Function to format nicely
        return formatted_result

    except Exception as e:
        return f"Error: {str(e)}"

# Transform pyodbc.Row objects into regular tuples so pandas can create the dataframe.
def process_database_result(result):
    data = [tuple(row) for row in result]  # Convert pyodbc.Row to tuples
    df = pd.DataFrame(data, columns=["PATIENT_ID", "NAME", "DATE_OF_BIRTH", 
                                     "DESCRIPTION", "TRANSCRIPTION", "KEYWORDS"])
   # desc_trnsc = df.iloc[3].to_string(index=False) + " " + df.iloc[4].to_string(index=False)
    return df

# PDF Handling - Ability to take a PDF, chunk it up and display portions or aks questions (Not working yet)
def handle_pdf_query(question, pdf_path):
    try:
        loader = DirectoryLoader(pdf_path)
        docs = loader.load()
        chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
        result = chain.run(docs, question)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Function to Generate Summary (Not working yet)
def generate_summary_with_LLM(text):
    response = OpenAI(temperature=0).call(text) 
    return response

def process_and_answer_question(question):
    try:
        cs=f'mssql+pyodbc://DESKTOP-AVM38KQ\MSSQLSERVER01/MED_DATA?driver=ODBC+Driver+17+for+SQL+Server'
        db_engine=create_engine(cs)
        db=SQLDatabase(db_engine)
        
        llm=ChatOpenAI(temperature=0.0,model="gpt-4")
        sql_toolkit=SQLDatabaseToolkit(db=db,llm=llm)
        sql_toolkit.get_tools()
        
        # Send OPENAI/ChatGPT a prompt describing the environment so it can more intelligently 
        # answer questions and build SQL. 
        prompt=ChatPromptTemplate.from_messages(
        [
        ("system",
        """
       You are a very intelligent AI assistant who is expert in identifying relevant questions from a user and converting 
       them into sql queries to generate the correct answer.
       Please use the below context to write the microsoft sql queries.
       context:
       You must query against the connected database, it has total 2 tables , these are PATIENT and PATIENT_NOTES .
       PATIENT table has columns PATIENT_ID, NAME, DATE_OF_BIRTH, GENDER, NEXT_APPT, PDF_AVAILABLE, PDF_DIRECTORY . 
       This table contains information about patients.
       PATIENT_NOTES table has columns PATIENT_ID, DESCRIPTION, TRANSCRIPTION, KEYWORDS, MEDICATIONS, ALLERGIES.
       This table contains detailed notes about the patient and the patients conditions .
       
       As an expert you must use joins whenever required.
        """
        ),
        ("user","{question}\ ai: ")
        ]
        )
        agent=create_sql_agent(llm=llm,toolkit=sql_toolkit,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True,max_execution_time=100,max_iterations=1000)
        # st.write(agent.run(prompt.format_prompt(question=question)))
        result = agent.run(prompt.format_prompt(question=question))
        # Check if we have a multiple items returned or just one.
        # If more than 1 data item returned, process the list otherwise process the string
        if isinstance(result, list):  
            for item in result:
                if isinstance(item, str):  
                    st.code(item)  
                else:
                    st.write(item)  
        else:
            st.write(result) 
    except Exception as e:
        return f"Error: {str(e)}"

# Set the title page
st.set_page_config(page_title="Generative AI Medical Assistant",page_icon=":speech_balloon:")
st.title("Generative AI Medical Assistant")

# Streamlit Sidebar for entering which request to make 
mode = st.sidebar.selectbox("Select Mode", ["Patient Detail", "Question", "PDF Analysis", "Exit"])
patient_id = st.sidebar.text_input("Patient ID")
question = st.sidebar.text_area("Enter your question:")
pdf_name = st.sidebar.text_input("PDF Name")

# LLM Summary
# st.subheader("Case Summary")
# summary = generate_summary_with_LLM(answer['DESCRIPTION'] + answer['TRANSCRIPTION'])
# st.write(summary)

# Streamlit logic flow
# First check the user selected prompt and then get the associate input
if st.sidebar.button("Submit"):
    if mode == "Patient Detail":
        if patient_id: 
            answer = handle_database_query(question, patient_id)
            st.write(answer)
        else:
            st.warning("Please enter a patient ID.")
    elif mode == "PDF Analysis":
        pdf_path = st.sidebar.text_input("PDF Name")  # Add PDF path input
        if pdf_path:
            answer = handle_pdf_query(question, pdf_path)
            st.write(answer)
        else:
            st.warning("Please enter a PDF document name.")
    elif mode == "Question":
        if question:
            process_and_answer_question(question)
            # st.write("Answer:", answer)
        else:
            st.warning("Problem answering question.")
    elif mode == "Exit":
        st.stop()