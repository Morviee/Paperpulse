import os
import streamlit as st
from langchain_groq import ChatGroq

def research_gap_node(state: dict) -> dict:
    """
    Research Gap Agent (finds gaps and unexplored areas)
    """
    print("-> [Research Gap Agent] Finding gaps and unexplored areas...")
    comparison = state.get("comparison", "")
    topic = state.get("topic", "AI")
    
    groq_api_key = os.environ.get("GROQ_API_KEY") or (st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None)
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing. Please add it to Streamlit Secrets.")
    
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.7, api_key=groq_api_key)
    
    prompt = f"""
    You are an AI research visionary focusing on '{topic}'.
    Based on the following comparison of recent literature:
    "{comparison}"
    
    Identify 5 highly detailed, specific "Research Gaps" or unexplored areas that future researchers should focus on.
    For each gap, provide a bolded title, a thorough explanation of why the gap exists based on the current literature, and potential steps to solve it.
    """
    
    response = llm.invoke(prompt)
    
    return {"research_gaps": response.content.strip()}
