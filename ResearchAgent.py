import re
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain_community.document_loaders import PyPDFLoader
import docx
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display

from FileHandler import llm, embeddings
from Nodes import doctor_agent, patient_agent, evaluationTool1, evaluationTool2, evaluationTool3, suggestion, reportGeneration,relevant, irrelevant
from testing_models import model1, model2

class State(MessagesState):
    relevant_questions: list
    irrelevant_questions: list
    doctor_questions:str
    patient_question: str

    patient_response: str
    doctor_response: str
    
    chat_context:list
    first_message: str
    relevant_answers: dict
    irrelevant_answers: dict
    
    evaluation: str
    report: str
    results: list
    resultT: str
    drawbacks: list
    suggestions: list
    system_info: str

# doctor_agent, patient_agent, evaluationTool, suggestion, reportGeneration = Nodes.doctor_agent(State), Nodes.patient_agent(State), Nodes.evaluationTool(State), Nodes.suggestion(State), Nodes.reportGeneration(State)   

os.getenv('PINECONE_API_KEY')
# groq_api = os.getenv('GROQ_API_KEY')

# llm = ChatGroq(groq_api_key=groq_api, model_name='Llama3-70b-8192', temperature=0.5)



def stop_condition(state: State):
    """Stop the graph"""
    if len(state["chat_context"])>=10:
        return "EvaluationTool3"
    else:
        return "BotUnderTest1"

build_graph = StateGraph(State)

build_graph.add_node("Act_as_End_User", relevant)
build_graph.add_node("TestsGenerator", irrelevant)
build_graph.add_node("BotUnderTest1", doctor_agent)
# build_graph.add_node("RAG", rag)
build_graph.add_node("BotUnderTest2", model1)
build_graph.add_node("BotUnderTest3", model2)
build_graph.add_node("ConversationHandler", patient_agent)
# build_graph.add_node("Model4", model)
build_graph.add_node("EvaluationTool1", evaluationTool1)
build_graph.add_node("EvaluationTool2", evaluationTool2)
build_graph.add_node("EvaluationTool3", evaluationTool3)
build_graph.add_node("Suggestion", suggestion)
build_graph.add_node("ReportGeneration", reportGeneration)

build_graph.add_edge(START, "Act_as_End_User")
build_graph.add_edge(START, "TestsGenerator")
build_graph.add_edge(START, "BotUnderTest1")
build_graph.add_edge("BotUnderTest1", "ConversationHandler")
build_graph.add_edge("Act_as_End_User", "BotUnderTest2")
build_graph.add_edge("TestsGenerator", "BotUnderTest3")
# build_graph.add_edge("RelevantQGeneratorO", "Model3")
build_graph.add_conditional_edges(
    "ConversationHandler",
    stop_condition
)

build_graph.add_edge("BotUnderTest2", "EvaluationTool1")
build_graph.add_edge("BotUnderTest3", "EvaluationTool2")
# build_graph.add_edge("Model3", "EvaluationTool")
# build_graph.add_edge("Model4", "EvaluationTool")
build_graph.add_edge("EvaluationTool1", "ReportGeneration")
build_graph.add_edge("EvaluationTool2", "ReportGeneration")
build_graph.add_edge("EvaluationTool3", "ReportGeneration")
build_graph.add_edge("ReportGeneration", "Suggestion")
build_graph.add_edge("Suggestion", END)

final_graph = build_graph.compile()
# display(Image(final_graph.get_graph(xray=True).draw_mermaid_png()))
png_bytes = final_graph.get_graph(xray=True).draw_mermaid_png()

# write to file
with open("my_graph2.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved to my_graph.png")


print("Graph Created")

# result = final_graph.invoke({"system_info":"Medical diagnosis chatbot to cure diseases and perscribe medicines"})
# result = final_graph.invoke({"system_info":"I have a headache and fever. Can you help me?",
#                              "chat_context": [], "results":[],
#                              "patient_response":"I have a headache and fever. Can you help me?"})

st.title("Chatbot Testing System")

system_info = st.text_input("System Info", "Medical Assistant trained to perscribe to treat patient diseases and any health related tips and excercises and perscribe medicines to patients.")
testing_agent = st.text_input("Testing Agent Role", "Patient")

if st.button("Start Testing"):
    with st.spinner("Processing queries..."):
        # Call the model function
        result = final_graph.invoke({"system_info": system_info,
                                     "chat_context": [], "results":[],
                                     "patient_response":"I have a headache and fever. Can you help me?",
                                     })
        
        with st.expander("See Full Generated Test Chat"):
            st.write(result["chat_context"])

        st.subheader("Evaluation Results")
        st.write(result["results"])
        st.write("__"*100)

        st.subheader("Evaluation Report")
        st.write(result["report"])
        st.write("__"*100)

        st.subheader("System Drawbacks and Improvement Suggestions")
        st.write(result["suggestions"])
        st.write("_-_"*100)