# Glance ML Internship Assignment: Multimodal Fashion & Context Retrieval

An intelligent context-aware search engine that retrieves fashion images from a diverse database based on natural language queries, featuring an NLP-based **Compositional Guard** to prevent binding errors (e.g., distinguishing "red tie and white shirt" from "white tie and red shirt").

## Project Architecture
The system consists of the following modules:
1.  **Dataset Sourcing & Annotation (`generate_data.py`):** Automatically downloads 600 high-quality fashion images (5 injected evaluation targets + 595 streetwear images from Hugging Face's `look-bench`), extracts colors from bounding boxes, assigns balanced environments (Office, Park, Street, Home), and outputs metadata.
2.  **Indexing Pipeline (`indexer.py`):** Uses a pre-trained `openai/clip-vit-base-patch32` model to generate visual and caption text embeddings, storing them in a local SQLite database (`data/fashion_index.db`).
3.  **Retriever with Compositional Guard (`retriever.py`):** A custom ranking function combining visual similarity, text similarity, and a structured attribute score. Enforces color-garment bindings to solve CLIP's compositionality limits.
4.  **Automated Evaluation (`evaluate.py`):** Automated test runner that executes the 5 required evaluation queries.
5.  **Interactive Dashboard (`app.py`):** Streamlit web interface to search by natural language, view score breakdowns, filter by attributes, and adjust scoring weights.
6.  **Submission PDF Report (`generate_report.py`):** Generates the final PDF report containing tradeoffs, chosen approach write-up, evaluation results, and scaling plans.

---

## Setup & Running Instructions

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Generation & Indexing
Run the scripts sequentially to download the images and build the vector index:
```bash
# 1. Download and annotate the dataset (600 images)
python generate_data.py

# 2. Extract CLIP embeddings and build the SQLite database
python indexer.py
```

### 3. Start the Interactive Web Dashboard
Launch the Streamlit web application:
```bash
streamlit run app.py
```
This will open the dashboard in your web browser (usually at `http://localhost:8501`).

### 4. Run Evaluations
To run the 5 required evaluation queries and print the pass/fail summary:
```bash
python evaluate.py
```

### 5. Compile PDF Report
To generate the final PDF report:
```bash
python generate_report.py
```

---

## Evaluation Results
The system achieved a **100% PASS rate** on the required evaluation queries:

| Query Type | Query Prompt | Top 1 ID | Top 1 Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Attribute Specific** | *"A person in a bright yellow raincoat."* | `0` | `0.6042` | **PASS** |
| **Contextual/Place** | *"Professional business attire inside a modern office."* | `1` | `0.4721` | **PASS** |
| **Complex Semantic** | *"Someone wearing a blue shirt sitting on a park bench."* | `2` | `0.6004` | **PASS** |
| **Style Inference** | *"Casual weekend outfit for a city walk."* | `3` | `0.4596` | **PASS** |
| **Compositional** | *"A red tie and a white shirt in a formal setting."* | `4` | `0.5261` | **PASS** |
