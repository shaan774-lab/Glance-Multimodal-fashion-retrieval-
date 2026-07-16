import streamlit as st
import os
import sqlite3
import pandas as pd
from PIL import Image
from retriever import FashionRetriever

# Page Configuration
st.set_page_config(
    page_title="Glance - Multimodal Fashion Search Engine",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .title-container {
        padding: 2rem;
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .title-container h1 {
        color: white !important;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .query-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }
    .image-card {
        background-color: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
        transition: transform 0.2s;
    }
    .image-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1);
    }
    .tag {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 0.25rem;
        margin-bottom: 0.25rem;
    }
    .tag-env { background-color: #d1fae5; color: #065f46; }
    .tag-type { background-color: #e0e7ff; color: #3730a3; }
    .tag-color { background-color: #fef3c7; color: #92400e; }
    .score-badge {
        font-size: 0.85rem;
        font-weight: 700;
        color: #4f46e5;
        background-color: #eeebff;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        float: right;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Retriever
@st.cache_resource
def get_retriever():
    return FashionRetriever()

retriever = get_retriever()

# Title banner
st.markdown("""
<div class="title-container">
    <h1>🛍️ Glance Multimodal Fashion Search Engine</h1>
    <p>Context-aware fashion retrieval using CLIP and Compositional Guard. Search by clothing, colors, location, and vibes.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("🔍 Retrieval Configuration")

st.sidebar.subheader("Score Weights")
w_visual = st.sidebar.slider("Visual Embeddings Weight", 0.0, 1.0, 0.5, 0.05)
w_textual = st.sidebar.slider("Textual Caption Weight", 0.0, 1.0, 0.2, 0.05)
w_attr = st.sidebar.slider("Structured Attribute Weight", 0.0, 1.0, 0.3, 0.05)

# Normalize weights
weight_sum = w_visual + w_textual + w_attr
if weight_sum > 0:
    w_visual /= weight_sum
    w_textual /= weight_sum
    w_attr /= weight_sum

st.sidebar.markdown(f"**Normalized Weights:**\n- Visual: `{w_visual:.2f}`\n- Textual: `{w_textual:.2f}`\n- Attr: `{w_attr:.2f}`")

st.sidebar.markdown("---")

# Quick search presets
st.sidebar.subheader("📋 Evaluation Presets")
presets = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting."
]

selected_preset = st.sidebar.radio("Click a preset query to run:", ["None"] + presets)

# Main query interface
st.subheader("🔎 Enter your query")
query_input = st.text_input(
    "Search the fashion collection...", 
    value="" if selected_preset == "None" else selected_preset,
    placeholder="e.g. Someone wearing a blue shirt sitting on a park bench."
)

k_val = st.slider("Number of results (k)", 1, 12, 6)

if query_input:
    st.markdown(f"### Results for: *\"{query_input}\"*")
    
    with st.spinner("Searching..."):
        results = retriever.search(query_input, top_k=k_val, w_visual=w_visual, w_textual=w_textual, w_attr=w_attr)
        
    if not results:
        st.warning("No matches found.")
    else:
        # Display parsed attributes
        parsed = retriever.parse_query(query_input)
        
        st.markdown("**Parsed Query Constraints:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"**Colors:** {', '.join(parsed['colors']) if parsed['colors'] else 'None'}")
        with col2:
            st.markdown(f"**Clothing:** {', '.join(parsed['categories']) if parsed['categories'] else 'None'}")
        with col3:
            st.markdown(f"**Environment:** {', '.join(parsed['environments']) if parsed['environments'] else 'None'}")
        with col4:
            st.markdown(f"**Styles:** {', '.join(parsed['styles']) if parsed['styles'] else 'None'}")
            
        if parsed["bindings"]:
            st.info(f"Detected color-clothing bindings: {', '.join([f'{c}-{cat}' for c, cat in parsed['bindings']])}")
            
        st.markdown("---")
        
        # Grid layout for results
        cols_per_row = 3
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(results):
                    res = results[i + j]
                    with cols[j]:
                        st.markdown(f"""
                        <div class="image-card">
                            <span class="score-badge">Score: {res['scores']['final']:.4f}</span>
                            <h4 style="margin-top:0;">Rank {i+j+1} (ID: {res['id']})</h4>
                        """, unsafe_allow_html=True)
                        
                        # Load and display image
                        if os.path.exists(res["image_path"]):
                            img = Image.open(res["image_path"])
                            st.image(img, use_column_width=True)
                        else:
                            st.error(f"Image not found: {res['image_path']}")
                            
                        # Show metadata tags
                        st.markdown(f"""
                            <div style="margin-top: 0.5rem; margin-bottom: 0.5rem;">
                                <span class="tag tag-env">{res['environment']}</span>
                                <span class="tag tag-type">{res['clothing_type']}</span>
                                <span class="tag tag-color">{res['color']}</span>
                            </div>
                            <p style="font-size: 0.85rem; font-style: italic; color: #555; height: 3em; overflow: hidden; text-overflow: ellipsis;">
                                "{res['caption']}"
                            </p>
                            <hr style="margin: 0.5rem 0;"/>
                            <div style="font-size: 0.8rem; color: #666;">
                                <b>Score Breakdown:</b><br/>
                                • Visual similarity: <code>{res['scores']['visual']:.4f}</code><br/>
                                • Textual similarity: <code>{res['scores']['textual']:.4f}</code><br/>
                                • Attribute match: <code>{res['scores']['attribute']:.4f}</code>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
else:
    # Default landing view showing database statistics
    st.markdown("### 📊 Database Statistics")
    
    conn = sqlite3.connect("data/fashion_index.db")
    df_stats = pd.read_sql_query("SELECT category, clothing_type, color, environment FROM fashion_items", conn)
    conn.close()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Indexed Images", len(df_stats))
    with col2:
        st.metric("Environments Represented", len(df_stats["environment"].unique()))
    with col3:
        st.metric("Unique Colors", len(df_stats["color"].unique()))
    with col4:
        st.metric("Unique Categories", len(df_stats["category"].unique()))
        
    st.markdown("---")
    
    st.markdown("#### Sample Environment Distribution")
    st.bar_chart(df_stats["environment"].value_counts())
    
    st.markdown("#### Sample Clothing Type Distribution")
    st.bar_chart(df_stats["clothing_type"].value_counts())
