from langchain_ollama import ChatOllama

def research_gap_node(state: dict) -> dict:
    """
    Research Gap Agent (finds gaps and unexplored areas)
    """
    print("-> [Research Gap Agent] Finding gaps and unexplored areas...")
    comparison = state.get("comparison", "")
    topic = state.get("topic", "AI")
    
    llm = ChatOllama(model="qwen2.5:0.5b", temperature=0.7)
    
    prompt = f"""
    You are an AI research visionary focusing on '{topic}'.
    Based on the following comparison of recent literature:
    "{comparison}"
    
    Identify 5 highly detailed, specific "Research Gaps" or unexplored areas that future researchers should focus on.
    For each gap, provide a bolded title, a thorough explanation of why the gap exists based on the current literature, and potential steps to solve it.
    """
    
    response = llm.invoke(prompt)
    
    return {"research_gaps": response.content.strip()}
