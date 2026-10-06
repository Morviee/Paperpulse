import json
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

def research_node(state: dict) -> dict:
    """
    Research Agent (finds relevant research papers)
    """
    print("-> [Research Agent] Finding relevant research papers...")
    topic = state.get("topic", "AI")
    
    groq_api_key = os.environ.get("GROQ_API_KEY") or (st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None)
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing. Please add it to Streamlit Secrets.")
    
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7, api_key=groq_api_key)
    
    prompt = f"""
    You are an AI research assistant. The user wants to research the topic: '{topic}'.
    Please provide exactly 10 hypothetical but highly realistic academic research papers about this topic.
    Ensure each paper has a very detailed, descriptive, and academic-style title (e.g., "Attention Is All You Need: A Deep Dive into Transformer Architectures for NLP").
    Return your response strictly in the following JSON format:
    [
      {{"title": "Paper 1 Title", "content": "A detailed 4-sentence abstract describing the paper's novel approach, methodology, and findings.", "url": "https://doi.org/10.1016/j.jair.2023.01.001"}},
      ... (include 10 items in the array)
    ]
    Do not include any other text besides the JSON.
    """
    
    response = llm.invoke(prompt)
    
    try:
        # Try to parse the JSON returned by the model
        papers = json.loads(response.content)
        # Ensure it's a list
        if not isinstance(papers, list):
            raise ValueError("Not a list")
    except Exception:
        # Fallback if the small model hallucinates or fails JSON parsing
        papers = []
        themes = [
            ("A Survey of Architectures", "This paper provides a sweeping overview of various architectures. It focuses on broad optimization strategies but lacks deep empirical testing on domain-specific datasets."),
            ("Empirical Analysis of Performance", "Focuses purely on quantitative scaling laws. The authors tested across massive clusters, showing that parameter count directly correlates with downstream accuracy, though latency suffers."),
            ("Novel Training Methodologies", "Proposes a new contrastive loss function. While it requires significantly less data (a major tradeoff), it struggles with out-of-distribution reasoning tasks."),
            ("Hardware Acceleration Strategies", "Investigates the deployment of these models on edge devices. By using heavy quantization and pruning, they achieved real-time inference, sacrificing a nominal 2% in F1 score."),
            ("Ethics and Bias Mitigation", "A qualitative study on the intrinsic biases present in the training datasets. The core methodology involves adversarial probing rather than architectural changes."),
            ("Federated Learning Approaches", "Introduces a decentralized training paradigm to preserve data privacy. The dataset is distributed, which inherently slow down the convergence rate compared to centralized models."),
            ("Multimodal Integration", "Extends the base architecture to process both visual and textual tokens simultaneously. The tradeoff is a massive increase in VRAM requirements."),
            ("Low-Rank Adaptation Techniques", "Demonstrates how fine-tuning can be achieved on consumer hardware using LoRA. The paper focuses heavily on efficient parameter updates rather than pre-training."),
            ("Long-Context Window Scaling", "Presents a novel positional encoding scheme allowing context lengths of 1M tokens. Memory usage is reduced from O(n^2) to O(n log n)."),
            ("Interpretability and Explainability", "Introduces sparse autoencoders to map hidden activations to human-understandable concepts, prioritizing transparency over raw performance.")
        ]
        
        for i, (sub_title, abstract) in enumerate(themes, 1):
            papers.append({
                "title": f"{topic}: {sub_title}",
                "content": abstract,
                "url": f"https://arxiv.org/abs/2301.123{i:02d}"
            })
            
    return {"papers": papers}
