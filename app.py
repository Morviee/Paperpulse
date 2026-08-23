import streamlit as st
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from agents.research_agent import research_node
from agents.analysis_agent import analysis_node
from agents.comparison_agent import comparison_node
from agents.research_gap_agent import research_gap_node

# Define the State for LangGraph
class GraphState(TypedDict):
    topic: str
    papers: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    comparison: str
    research_gaps: str

def build_graph():
    # Initialize the Graph
    workflow = StateGraph(GraphState)
    
    # Add Nodes
    workflow.add_node("research", research_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("comparison", comparison_node)
    workflow.add_node("research_gap", research_gap_node)
    
    # Add Edges to create a sequential pipeline
    workflow.set_entry_point("research")
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "comparison")
    workflow.add_edge("comparison", "research_gap")
    workflow.add_edge("research_gap", END)
    
    # Compile Graph
    return workflow.compile()

def main():
    st.set_page_config(page_title="PaperPulse - AI Research Detective", page_icon="💡", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for UI styling
    st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: 0.3s;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .paper-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #2e86de;
    }
    .paper-card h3 {
        color: #2e86de;
        margin-top: 0;
    }
    .paper-card a {
        display: inline-block;
        margin-top: 10px;
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 5px 10px;
        border-radius: 5px;
        text-decoration: none;
        font-size: 14px;
        font-weight: 500;
    }
    .paper-card a:hover {
        background-color: #bae6fd;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3145/3145765.png", width=100)
        st.title("PaperPulse 📚")
        st.markdown("Your **AI-powered** research assistant. Find, analyze, and compare papers effortlessly.")
        st.divider()
        st.subheader("🔍 New Research")
        topic = st.text_input("Enter Topic:", "Large Language Models", help="E.g., Quantum Computing, Vision Transformers")
        run_btn = st.button("🚀 Start Analysis", type="primary")
        
        st.divider()
        st.info("💡 **Tip**: Be specific with your topic for better results.")
    
    # Main Canvas
    st.title("🔬 Research Dashboard")
    st.markdown("Automated AI pipeline to discover insights, compare literature, and uncover research gaps.")
    
    if run_btn:
        with st.status(f"🧠 Running AI Pipeline on '{topic}'...", expanded=True) as status:
            st.write("Initializing LangGraph workflow...")
            app = build_graph()
            initial_state = {"topic": topic}
            
            try:
                st.write("Invoking AI agents (Research, Analysis, Comparison, Gaps)...")
                result = app.invoke(initial_state)
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="❌ Pipeline Failed", state="error")
                st.error(f"An error occurred: {e}")
                result = None
                
        if result:
            st.write("---")
            st.header(f"Results for: `{topic}`")
            
            # Metrics Row
            col1, col2, col3 = st.columns(3)
            papers = result.get("papers", [])
            
            col1.metric("Papers Analyzed", len(papers))
            col2.metric("Research Gaps Identified", len(result.get('research_gaps', ' ').split('. ')) if result.get('research_gaps') else 0)
            col3.metric("Analysis Confidence", "High")
            
            st.write("---")
            
            # Tabs for detailed structured format
            tab1, tab2, tab3, tab4 = st.tabs(["📑 Papers Found", "📊 Deep Analysis", "⚖️ Comparison", "🎯 Research Gaps"])
            
            with tab1:
                for i, p in enumerate(papers, 1):
                    title = p.get('title', f'Paper {i}')
                    content = p.get('content', 'No abstract available.')
                    url = p.get('url', '#')
                    
                    st.markdown(f"""
                    <div class="paper-card">
                        <h3>{title}</h3>
                        <p>{content}</p>
                        <a href="{url}" target="_blank">🔗 Read Full Paper</a>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with tab2:
                analysis = result.get("analysis", [])
                if analysis and isinstance(analysis, list):
                    for item in analysis:
                        with st.expander(f"🔍 {item.get('title', 'Unknown Paper')}", expanded=True):
                            st.markdown(f"**❖ Methodology**\\n{item.get('methodology', 'N/A')}")
                            st.markdown(f"**❖ Datasets**\\n{item.get('datasets', 'N/A')}")
                            st.markdown(f"**❖ Results**\\n{item.get('results', 'N/A')}")
                elif analysis and isinstance(analysis, dict):
                    # Legacy fallback just in case
                    for k, v in analysis.items():
                        with st.container():
                            st.markdown(f"### ❖ {k.replace('_', ' ').title()}")
                            st.info(v)
                else:
                    st.warning("No structural analysis available.")
                    
            with tab3:
                st.markdown("### ⚖️ Literature Comparison")
                comp_text = result.get('comparison', 'No comparison available.')
                st.markdown(comp_text)
                
            with tab4:
                st.markdown("### 🎯 Uncharted Territories (Gaps)")
                gap_text = result.get('research_gaps', 'No research gaps available.')
                st.markdown(gap_text)
                    
    else:
        # Initial Placeholder View
        st.write("👈 Start a research journey by entering a topic in the sidebar.")
        st.image("https://undraw.co/api/illustrations/svg/undraw_researching_22gp")

if __name__ == "__main__":
    main()
