from langchain_ollama import ChatOllama

def comparison_node(state: dict) -> dict:
    """
    Comparison Agent (compares papers and detects contradictions)
    """
    print("-> [Comparison Agent] Comparing papers and detecting contradictions...")
    analysis = state.get("analysis", [])
    topic = state.get("topic", "AI")
    papers = state.get("papers", [])
    
    llm = ChatOllama(model="qwen2.5:0.5b", temperature=0.5)
    
    paper_texts = "\\n".join([f"- {p['title']}: {p['content']}" for p in papers])
    
    # Format analysis context depending on if it's a list (new format) or dict (old format)
    if isinstance(analysis, list):
        analysis_context = "\\n".join([f"- {a.get('title', 'Paper')}\\n  Methodology: {a.get('methodology', '')}\\n  Datasets: {a.get('datasets', '')}\\n  Results: {a.get('results', '')}" for a in analysis])
    else:
        analysis_context = f"Methodology: {analysis.get('methodology')}\\nDatasets: {analysis.get('datasets')}\\nResults: {analysis.get('results')}"

    prompt = f"""
    You are an AI research expert looking at research in '{topic}'.
    Based on these papers:
    {paper_texts}
    
    And this generalized analysis:
    {analysis_context}
    
    Write a comprehensive comparison of these papers. 
    You MUST output your response STRICTLY as a detailed Markdown table with the following columns:
    | Paper Title | Core Methodology | Dataset / Scale | Key Finding / Tradeoff |
    
    Make sure to compare all the provided papers and format the output beautifully in Markdown. Do not include extra text outside the table.
    """
    
    response = llm.invoke(prompt)
    
    return {"comparison": response.content.strip()}
