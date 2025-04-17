import time
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain.chains import RetrievalQA
import streamlit as st

groq_api = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key=groq_api, model_name='Llama3-70b-8192', temperature=0.5)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

knowledge_base = PineconeVectorStore.from_existing_index(
        index_name="medicalbot", embedding=embeddings
    )

qa = RetrievalQA.from_chain_type(
      llm=llm, 
      chain_type="stuff",
      retriever=knowledge_base.as_retriever(search_kwargs={"k": 2}),
      return_source_documents=True,
      # chain_type_kwargs=chain_type_kwargs,
  )


def model1(state):
    """Model to be tested"""
    questions = state["relevant_questions"]
    generatedData = {}
    for user_input in questions:
        user_inpt= f"""You are doctor . trained to only answer and perscribe patients about diseases. Do not provide any other information  even 
        it is medical related. Only stick in perscribing patient diseases.This is user query{user_input}.
        If user query is not about medical or health related, then only say "I am not able to help you with this query".
        If user query is about medical or health related disease or treatement or medicine, then only respond with suitable answer.
        Answer shortly and precisely.
        """
        result = qa({"query": user_inpt})
        ans = result['result']
        print("User Input: ", user_input)
        print(ans)
        with st.expander("Model 1 Response"):
            st.write("User Input: ", user_input)
            st.write(ans)
        print("-"*50)
        generatedData[user_input] = ans
        time.sleep(60)
        
    print("*"*100)
    # state["chat_context"].append({"role": "user", "content": user_input})
    # state["chat_context"].append({"role": "doctor", "content": ans})
    return {"relevant_answers": generatedData}



def model2(state):
    """Model to be tested"""
    questions = state["irrelevant_questions"]
    generatedData = {}
    for user_input in questions:
        user_inpt= f"""
        You are doctor . trained to only answer and perscribe patients about diseases. Do not provide any other information  even 
        it is medical related. Only stick in perscribing patient diseases.
        This is user query{user_input}
        If user query is not about medical or health related, then only say "I am not able to help you with this query".
        If user query is about medical or health related disease or treatement or medicine, then only respond with suitable answer.
        Answer shortly and precisely.
        """
        result = qa({"query": user_inpt})
        ans = result['result']
        print("User Input: ", user_input)
        print(ans)
        with st.expander("Model 2 Response"):
            st.write("User Input: ", user_input)
            st.write(ans)
        print("-"*50)
        generatedData[user_input] = ans
        time.sleep(60)
        
    print("*"*100)
    # state["chat_context"].append({"role": "user", "content": user_input})
    # state["chat_context"].append({"role": "doctor", "content": ans})
    return {"irrelevant_answers": generatedData}

print("Models Created")