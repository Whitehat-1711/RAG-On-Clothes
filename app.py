import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient

# -----------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------
st.set_page_config(
    page_title="Fashion Vibe Matcher",
    page_icon="👗",
    layout="wide"
)

# -----------------------------------------------
# Custom CSS Styling
# -----------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B9D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .product-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .score-excellent {
        color: #00ff00;
        font-weight: bold;
    }
    .score-good {
        color: #ffff00;
        font-weight: bold;
    }
    .score-weak {
        color: #ff0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------
# Initialize Session State
# -----------------------------------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'embeddings_generated' not in st.session_state:
    st.session_state.embeddings_generated = False
if 'hf_token' not in st.session_state:
    st.session_state.hf_token = None


# -----------------------------------------------
# Hugging Face Inference Function
# -----------------------------------------------
def get_embedding_from_hf(
    text: str,
    model_name: str,
    api_token: str,
    max_retries: int = 3
):
    """
    Generate sentence embeddings using Hugging Face InferenceClient API.
    Automatically normalizes embeddings and retries on failure.
    """
    client = InferenceClient(model=model_name, token=api_token)

    for attempt in range(max_retries):
        try:
            # Feature extraction API returns list of embeddings
            response = client.feature_extraction(
                text,
            )

            embedding = np.array(response)
            if embedding.ndim > 1:
                embedding = embedding[0]

            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm == 0:
                st.error("Received zero-vector embedding from Hugging Face API")
                return None
            return embedding / norm

        except Exception as e:
            st.warning(f"⚠️ Attempt {attempt + 1}/{max_retries} failed: {e}")
            time.sleep(2 * (attempt + 1))
            continue

    st.error("❌ Failed to get embedding from Hugging Face API after multiple retries.")
    return None


# -----------------------------------------------
# Load CSV Data
# -----------------------------------------------
@st.cache_data
def load_data(uploaded_file):
    """Load and prepare the dataset"""
    df = pd.read_csv(uploaded_file)
    df['combined_text'] = df['name'] + ": " + df['desc'] + " (" + df['vibes'] + ")"
    return df


# -----------------------------------------------
# Generate Embeddings
# -----------------------------------------------
def generate_embeddings(df, model_name: str, api_token: str):
    """Generate embeddings for all products using HF API"""
    embeddings_list = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, text in enumerate(df['combined_text']):
        embedding = get_embedding_from_hf(text, model_name, api_token)
        embeddings_list.append(embedding)

        progress = (idx + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Processing: {df['name'].iloc[idx]} ({idx + 1}/{len(df)})")

        # Small delay to avoid hitting API limits
        time.sleep(0.3)

    progress_bar.empty()
    status_text.empty()
    df['embedding'] = embeddings_list
    return df


# -----------------------------------------------
# Search Function
# -----------------------------------------------
def search_products(query: str, df: pd.DataFrame, model_name: str, api_token: str, top_k: int = 3):
    """Search for products matching the query"""
    query_embedding = get_embedding_from_hf(query, model_name, api_token)

    if query_embedding is None:
        st.error("Failed to generate query embedding")
        return None, 0

    start_time = time.time()

    # Calculate cosine similarity
    similarities = []
    for idx, prod_embedding in enumerate(df['embedding']):
        if prod_embedding is not None:
            similarity = cosine_similarity([query_embedding], [prod_embedding])[0][0]
            similarities.append({
                'index': idx,
                'name': df['name'].iloc[idx],
                'description': df['desc'].iloc[idx],
                'vibes': df['vibes'].iloc[idx],
                'similarity_score': similarity
            })

    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    query_time = time.time() - start_time

    return similarities[:top_k], query_time


# -----------------------------------------------
# Main Streamlit App UI
# -----------------------------------------------
st.markdown('<h1 class="main-header">👗 Fashion Vibe Matcher 🎨</h1>', unsafe_allow_html=True)
st.markdown("""
### Welcome to the Fashion Vibe Matcher!
Upload your fashion dataset and search for products that match your desired vibe using **AI-powered semantic search** powered by Hugging Face 🤗.
""")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # HuggingFace API Token
    st.header("🔑 API Configuration")
    hf_token = st.text_input(
        "HuggingFace API Token",
        type="password",
        help="Get your free token at https://huggingface.co/settings/tokens"
    )

    if hf_token:
        st.session_state.hf_token = hf_token
        st.success("✅ API token set!")
    else:
        st.warning("⚠️ Please enter your HuggingFace API token to continue")

    # Model selection
    model_choice = st.selectbox(
        "Select Embedding Model",
        [
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-MiniLM-L3-v2"
        ],
        index=1
    )

    # File upload
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV should have columns: name, desc, vibes"
    )

    if uploaded_file is not None and st.session_state.hf_token:
        if st.button("🚀 Load Data & Generate Embeddings"):
            # Load data
            st.session_state.df = load_data(uploaded_file)
            st.success(f"✅ Loaded {len(st.session_state.df)} products")

            # Generate embeddings
            with st.spinner("Generating embeddings via HuggingFace Inference API..."):
                st.session_state.df = generate_embeddings(
                    st.session_state.df,
                    model_choice,
                    st.session_state.hf_token
                )
            st.session_state.embeddings_generated = True
            st.success("✅ Embeddings generated successfully!")
            st.balloons()

    # Display dataset info
    if st.session_state.df is not None:
        st.header("📊 Dataset Info")
        st.metric("Total Products", len(st.session_state.df))
        st.metric("Model", model_choice.split('/')[-1])


# -----------------------------------------------
# Search Interface
# -----------------------------------------------
if st.session_state.embeddings_generated and st.session_state.hf_token:
    st.header("🔍 Search for Your Perfect Vibe")

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input(
            "Describe your desired style:",
            placeholder="e.g., urban streetwear for confident city style",
            help="Enter a description of the fashion vibe you're looking for"
        )
    with col2:
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)

    if query:
        with st.spinner("Searching for matches..."):
            results, query_time = search_products(
                query,
                st.session_state.df,
                model_choice,
                st.session_state.hf_token,
                top_k
            )

        if results:
            st.success(f"⏱️ Found {len(results)} matches in {query_time:.3f} seconds")

            st.header("🏆 Top Matches")
            for i, result in enumerate(results, 1):
                score = result['similarity_score']
                if score > 0.7:
                    quality = "🟢 Excellent Match"
                    score_class = "score-excellent"
                elif score > 0.5:
                    quality = "🟡 Good Match"
                    score_class = "score-good"
                else:
                    quality = "🔴 Weak Match"
                    score_class = "score-weak"

                with st.expander(f"**Rank {i}: {result['name']}** - {quality}", expanded=(i==1)):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Description:** {result['description']}")
                        st.markdown(f"**Vibes:** `{result['vibes']}`")
                    with col2:
                        st.markdown(f"**Similarity Score**")
                        st.markdown(f"<p class='{score_class}'>{score:.4f}</p>", unsafe_allow_html=True)
                        st.progress(float(min(max(score, 0), 1)))

            if results[0]['similarity_score'] < 0.5:
                st.warning("""
                ⚠️ **No strong matches found.** Try refining your search query or using more descriptive text.
                """)

else:
    st.info("""
    ### 📋 Getting Started:

    1. **Get a HuggingFace API token** (free):
       - Go to https://huggingface.co/settings/tokens
       - Create a new token (read access is enough)
       - Paste it in the sidebar

    2. **Upload your CSV file** in the sidebar  
       Required columns: `name`, `desc`, `vibes`

    3. **Select an embedding model**

    4. **Click "Load Data & Generate Embeddings"**

    5. **Start searching** for your perfect fashion vibe!
    """)

    with st.expander("📄 View Sample Dataset Format"):
        sample_data = {
            'name': ['Boho Dress', 'Street Hoodie', 'Classic Blazer'],
            'desc': [
                'Flowy, earthy tones for festival vibes',
                'Oversized hoodie with graphic print',
                'Tailored blazer for formal appearance'
            ],
            'vibes': [
                'boho, relaxed, nature',
                'urban, casual, cool',
                'formal, elegant, classic'
            ]
        }
        st.dataframe(pd.DataFrame(sample_data))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Built with ❤️ using Streamlit & HuggingFace Inference API
</div>
""", unsafe_allow_html=True)
