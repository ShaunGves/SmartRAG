"""
ui/app.py

Streamlit chat interface for the SmartRAG pipeline.

Features:
  - Chat history with session state
  - Source document display
  - Performance metrics display
  - Document ingestion sidebar
  - Evaluation dashboard with ablation study results

Run: streamlit run ui/app.py
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartRAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; color: #ffffff; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #2a2d3a 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border-left: 5px solid #00d4aa;
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.1);
        margin: 0.5rem 0;
    }
    .source-box {
        background: #1a1d2e;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        border-left: 4px solid #2196F3;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.1);
    }
    .stChatMessage { 
        border-radius: 15px; 
        background: #1e2130;
        border: 1px solid #2a2d3a;
    }
    .tab-content {
        background: #0e1117;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .eval-metric {
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        color: #00d4aa;
    }
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #1e2130;
        border-radius: 10px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: #ffffff;
        background: transparent;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #00d4aa;
        color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "avg_latency" not in st.session_state:
    st.session_state.avg_latency = 0.0

# ─── Header ───────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🧠 SmartRAG")
    st.caption("Fine-tuned LLM + Retrieval-Augmented Generation")

# ─── Tabs ─────────────────────────────────────────────────────────
tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Control Panel</div>', unsafe_allow_html=True)

    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 4)
    st.divider()

    # Health check
    st.header("📡 System Status")
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=3)
        health = resp.json()
        st.success("API Online" if health["status"] == "healthy" else "API Degraded")
        st.write(f"Model: {'✅' if health['model_loaded'] else '❌'}")
        st.write(f"VectorStore: {'✅' if health['vectorstore_loaded'] else '❌'}")
    except Exception:
        st.error("API Offline — start with: `uvicorn api.app:app`")

    st.divider()

    # Document ingestion
    st.header("📄 Add Documents")
    uploaded = st.file_uploader("Upload .txt file", type=["txt"])
    if uploaded and st.button("Ingest Document"):
        text = uploaded.read().decode("utf-8")
        with st.spinner("Ingesting..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/ingest",
                    json={"texts": [text], "metadata": [{"filename": uploaded.name}]},
                )
                st.success(f"✅ Ingested: {uploaded.name}")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ─── Chat Tab ─────────────────────────────────────────────────────
with tab_chat:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    # Chat Interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source(s) used"):
                    for src in msg["sources"]:
                        st.markdown(f'<div class="source-box">{src}</div>', unsafe_allow_html=True)
            if msg.get("latency_ms"):
                st.caption(f"⚡ {msg['latency_ms']:.0f}ms · {msg.get('num_chunks', 0)} chunks retrieved")

    # Input
    if prompt := st.chat_input("Ask anything about your documents..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/query",
                        json={"question": prompt, "top_k": top_k},
                        timeout=60,
                    )
                    data = resp.json()

                    answer = data.get("answer", "No answer returned.")
                    sources = data.get("sources", [])
                    latency = data.get("latency_ms", 0)
                    num_chunks = data.get("num_chunks_retrieved", 0)

                    st.write(answer)

                    if sources:
                        with st.expander(f"📎 {len(sources)} source(s) used"):
                            for src in sources:
                                st.markdown(f'<div class="source-box">{src}</div>', unsafe_allow_html=True)

                    st.caption(f"⚡ {latency:.0f}ms · {num_chunks} chunks retrieved")

                    # Update session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "latency_ms": latency,
                        "num_chunks": num_chunks,
                    })
                    st.session_state.total_queries += 1

                    # Update running avg latency
                    n = st.session_state.total_queries
                    st.session_state.avg_latency = (
                        (st.session_state.avg_latency * (n - 1) + latency) / n
                    )

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Run: `uvicorn api.app:app --port 8000`")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Session Stats
    st.divider()
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.metric("Total Queries", st.session_state.total_queries)
    with col2:
        st.metric("Avg Latency", f"{st.session_state.avg_latency:.0f} ms")
    with col3:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Evaluation Tab ───────────────────────────────────────────────
with tab_eval:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    
    st.header("🔬 Ablation Study Results")
    st.caption("Comparing different pipeline configurations")
    
    # Load evaluation data
    try:
        # Load summary
        with open(Path(__file__).parent.parent / "artifacts" / "eval_results" / "ablation_summary.json", "r") as f:
            summary = json.load(f)
        
        # Load detailed results
        df = pd.read_csv(Path(__file__).parent.parent / "artifacts" / "eval_results" / "ablation_results.csv")
        
        # System names mapping
        system_names = {
            "A_base_dense": "Base Model + Dense RAG",
            "B_finetuned_dense": "Fine-Tuned + Dense RAG", 
            "C_finetuned_hybrid": "Fine-Tuned + Hybrid Search",
            "D_full_pipeline": "Fine-Tuned + Hybrid + Reranking"
        }
        
        # Metrics overview
        st.subheader("📈 Performance Metrics")
        cols = st.columns(4)
        for i, (key, data) in enumerate(summary.items()):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{system_names.get(key, key)}</h4>
                    <div class="eval-metric">{data['avg_faithfulness']:.2f}</div>
                    <small>Faithfulness</small><br>
                    <div class="eval-metric">{data['avg_keyword_coverage']:.2f}</div>
                    <small>Keyword Coverage</small><br>
                    <div class="eval-metric">{data['avg_latency_ms']:.0f}ms</div>
                    <small>Avg Latency</small><br>
                    <div class="eval-metric">{data['failure_rate']:.1%}</div>
                    <small>Failure Rate</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Charts
        st.subheader("📊 Comparative Analysis")
        
        # Faithfulness chart
        fig_faith = px.bar(
            x=[system_names[k] for k in summary.keys()],
            y=[v['avg_faithfulness'] for v in summary.values()],
            title="Faithfulness Score by Pipeline",
            labels={'x': 'Pipeline', 'y': 'Faithfulness'},
            color=[v['avg_faithfulness'] for v in summary.values()],
            color_continuous_scale='Viridis'
        )
        fig_faith.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='#ffffff'
        )
        st.plotly_chart(fig_faith, use_container_width=True)
        
        # Latency chart
        fig_lat = px.bar(
            x=[system_names[k] for k in summary.keys()],
            y=[v['avg_latency_ms'] for v in summary.values()],
            title="Average Latency by Pipeline",
            labels={'x': 'Pipeline', 'y': 'Latency (ms)'},
            color=[v['avg_latency_ms'] for v in summary.values()],
            color_continuous_scale='Plasma'
        )
        fig_lat.update_layout(
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117',
            font_color='#ffffff'
        )
        st.plotly_chart(fig_lat, use_container_width=True)
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        st.dataframe(df, use_container_width=True)
        
        # Failure analysis
        st.subheader("❌ Failure Analysis")
        failure_df = df[df['failure'] != 'none']
        if not failure_df.empty:
            st.dataframe(failure_df[['system', 'question', 'failure']], use_container_width=True)
        else:
            st.success("No failures detected in the evaluation set!")
            
    except FileNotFoundError:
        st.warning("Evaluation results not found. Run the ablation study first: `python -m evaluation.ablation`")
    except Exception as e:
        st.error(f"Error loading evaluation data: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)
