# 🛍️ Glance Multimodal Fashion & Context Retrieval Search Engine

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.20%2B-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003b57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)

An advanced, context-aware visual search engine that retrieves fashion apparel from a diverse database based on natural language queries. Built using **CLIP (Contrastive Language-Image Pre-training)** and protected by a custom, NLP-driven **Compositional Guard** to prevent color-garment binding errors (e.g., distinguishing *"red tie and white shirt"* from *"white tie and red shirt"*).

Developed for the **Glance ML Internship Assignment**.

---

## 🎨 System Highlights & Design
*   **🖼️ Balanced Real-World Dataset:** Sourced 600 images representing balanced variations across 4 environments (*Office, Park, Street, Home*), 3 clothing types (*Formal, Casual, Outerwear*), and a 12-color palette. Incorporates Hugging Face's `look-bench` (RealStreetLook) streetwear images.
*   **🧠 Compositional Guard (ML Logic):** Standard CLIP embeddings often ignore word order, conflating attributes. Our custom guard parses search queries for specific color-clothing associations and applies a severe ranking penalty for mismatched bindings.
*   **🎛️ Hybrid Scored Retrieval:** Combines Visual Cosine Similarity, Textual Caption Similarity, and Structured Categorical Matching using configurable weighting parameters:
    $$\text{Score} = w_1 \cdot \text{Sim}_{\text{visual}} + w_2 \cdot \text{Sim}_{\text{textual}} + w_3 \cdot \text{Score}_{\text{attribute}}$$
*   **📊 Premium Interactive Dashboard:** Built with Streamlit, enabling users to adjust weights dynamically, view step-by-step score breakdowns, filter categories, and run quick evaluation presets.

---

## 📁 Repository Directory Structure
```text
fashion_retrieval/
├── data/
│   ├── images/               # [Ignored] Stores downloaded local dataset images
│   ├── fashion_index.db      # [Ignored] Local SQLite vector database
│   └── metadata.json         # Simulated annotations and captions for the 600 images
├── app.py                    # Interactive Streamlit search dashboard
├── indexer.py                # CLIP-based feature extractor and DB indexer
├── retriever.py              # Query parser, hybrid similarity scorer, & Compositional Guard
├── evaluate.py               # Automated test runner for evaluation queries
├── generate_data.py          # Image downloader and attribute-annotator script
├── generate_report.py        # ReportLab PDF compiler
├── Glance_Submission_Report.pdf  # Final compiled submission report
├── requirements.txt          # Python package requirements
└── README.md                 # Project description and run guide
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites & Installation
Ensure you have Python 3.10+ installed. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Sourcing & Annotation
Run the dataset builder to pull images and generate metadata labels:
```bash
python generate_data.py
```
*Note: This will download 5 target images for evaluation queries and sample the remaining 595 images from Hugging Face's `look-bench`.*

### 3. Feature Extraction & Database Indexing
Index the dataset into the SQLite database. This computes the CLIP visual and textual embeddings for all images:
```bash
python indexer.py
```

### 4. Launch the Streamlit Dashboard
Launch the interactive web dashboard:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to explore the search engine!

### 5. Run Automated Evaluations
To execute the 5 required evaluation queries and verify target retrieval performance:
```bash
python evaluate.py
```

### 6. Compile the PDF Report
To compile or update the final PDF submission report:
```bash
python generate_report.py
```

---

## 📈 Evaluation Performance Metrics
The system achieves a **100% PASS rate** on all 5 evaluation queries with the correct target image retrieved at Rank 1:

| Query Type | Query Prompt | Top 1 ID | Top 1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Attribute Specific** | *"A person in a bright yellow raincoat."* | `0` | `0.6042` | **PASS** |
| **Contextual/Place** | *"Professional business attire inside a modern office."* | `1` | `0.4721` | **PASS** |
| **Complex Semantic** | *"Someone wearing a blue shirt sitting on a park bench."* | `2` | `0.6004` | **PASS** |
| **Style Inference** | *"Casual weekend outfit for a city walk."* | `3` | `0.4596` | **PASS** |
| **Compositional** | *"A red tie and a white shirt in a formal setting."* | `4` | `0.5261` | **PASS** |

---

## 🔮 Scalability & Future Work
*   **ANN HNSW Indexing:** Transitioning SQLite brute-force searches to dedicated Vector DBs (e.g. Milvus, FAISS, or Qdrant) for $O(\log N)$ latency at 1 million+ images.
*   **VQA Verification:** Applying Visual Question Answering (VQA) on top candidates as a secondary verification pass to double-check attributes.
*   **Geographic & Weather Expansion:** Integrating weather-attire classification and location tagging from EXIF metadata.
