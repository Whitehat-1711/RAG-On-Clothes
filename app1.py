import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="Fashion Vibe Matcher",
    page_icon="👗",
    layout="wide"
)

# Custom CSS
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

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'embeddings_generated' not in st.session_state:
    st.session_state.embeddings_generated = False

@st.cache_resource
def load_model(model_choice):
    """Load the sentence transformer model"""
    with st.spinner(f"Loading {model_choice} model..."):
        model = SentenceTransformer(model_choice)
    return model

@st.cache_data
def load_data(uploaded_file):
    """Load and prepare the dataset"""
    df = pd.read_csv(uploaded_file)
    df['combined_text'] = df['name'] + ": " + df['desc'] + " (" + df['vibes'] + ")"
    return df

def generate_embeddings(df, model):
    """Generate embeddings for all products"""
    embeddings_list = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, text in enumerate(df['combined_text']):
        try:
            embedding = model.encode(text, normalize_embeddings=True)
            embeddings_list.append(embedding)
            progress = (idx + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {df['name'].iloc[idx]} ({idx + 1}/{len(df)})")
        except Exception as e:
            st.error(f"Error for {df['name'].iloc[idx]}: {str(e)}")
            embeddings_list.append(None)
    
    progress_bar.empty()
    status_text.empty()
    df['embedding'] = embeddings_list
    return df

def search_products(query: str, df: pd.DataFrame, model, top_k: int = 3):
    """Search for products matching the query"""
    try:
        query_embedding = model.encode(query, normalize_embeddings=True)
    except Exception as e:
        st.error(f"Error generating query embedding: {str(e)}")
        return None, 0
    
    start_time = time.time()
    
    # Calculate cosine similarity
    similarities = []
    for idx, prod_embedding in enumerate(df['embedding']):
        if prod_embedding is not None:
            similarity = cosine_similarity(
                [query_embedding],
                [prod_embedding]
            )[0][0]
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

# Main App
st.markdown('<h1 class="main-header">👗 Fashion Vibe Matcher 🎨</h1>', unsafe_allow_html=True)

st.markdown("""
### Welcome to the Fashion Vibe Matcher!
Upload your fashion dataset and search for products that match your desired vibe using AI-powered semantic search.
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    model_choice = st.selectbox(
        "Select Embedding Model",
        [
            "all-mpnet-base-v2",
            "all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L3-v2"
        ],
        index=0
    )
    
    # File upload
    st.header("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV should have columns: name, desc, vibes"
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Load Data & Generate Embeddings"):
            # Load model
            st.session_state.model = load_model(model_choice)
            
            # Load data
            st.session_state.df = load_data(uploaded_file)
            st.success(f"✅ Loaded {len(st.session_state.df)} products")
            
            # Generate embeddings
            with st.spinner("Generating embeddings..."):
                st.session_state.df = generate_embeddings(
                    st.session_state.df, 
                    st.session_state.model
                )
            st.session_state.embeddings_generated = True
            st.success("✅ Embeddings generated successfully!")
            st.balloons()
    
    # Display dataset info
    if st.session_state.df is not None:
        st.header("📊 Dataset Info")
        st.metric("Total Products", len(st.session_state.df))
        st.metric("Model", model_choice)

# Main content
if st.session_state.embeddings_generated:
    st.header("🔍 Search for Your Perfect Vibe")
    
    # Create two columns
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
                st.session_state.model, 
                top_k
            )
        
        if results:
            st.success(f"⏱️ Found {len(results)} matches in {query_time:.3f} seconds")
            
            # Display results
            st.header("🏆 Top Matches")
            
            for i, result in enumerate(results, 1):
                score = result['similarity_score']
                
                # Determine quality
                if score > 0.7:
                    quality = "🟢 Excellent Match"
                    score_class = "score-excellent"
                elif score > 0.5:
                    quality = "🟡 Good Match"
                    score_class = "score-good"
                else:
                    quality = "🔴 Weak Match"
                    score_class = "score-weak"
                
                # Create expandable card
                with st.expander(f"**Rank {i}: {result['name']}** - {quality}", expanded=(i==1)):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {result['description']}")
                        st.markdown(f"**Vibes:** `{result['vibes']}`")
                    
                    with col2:
                        st.markdown(f"**Similarity Score**")
                        st.markdown(f"<p class='{score_class}'>{score:.4f}</p>", unsafe_allow_html=True)
                        
                        # Progress bar for score
                        st.progress(score)
            
            # Warning for weak matches
            if results[0]['similarity_score'] < 0.5:
                st.warning("""
                ⚠️ **No strong matches found.** Consider:
                - Refining your search query
                - Adding more products to the catalog
                - Using different keywords
                """)
    
    # Example queries
    with st.expander("💡 Try these example queries"):
        example_queries = [
            "urban streetwear for confident city style",
            "comfortable sunny-day outfit with carefree energy",
            "sleek minimal street-style jacket",
            "cozy winter outfit for relaxed evenings",
            "elegant formal dress for special occasions"
        ]
        
        for example in example_queries:
            if st.button(example, key=example):
                st.session_state.example_query = example
                st.rerun()

else:
    # Instructions
    st.info("""
    ### 📋 Getting Started:
    
    1. **Upload your CSV file** in the sidebar
       - Required columns: `name`, `desc`, `vibes`
    
    2. **Select an embedding model**
       - `all-mpnet-base-v2`: Best quality (recommended)
       - `all-MiniLM-L6-v2`: Faster, good quality
       - `paraphrase-MiniLM-L3-v2`: Fastest, decent quality
    
    3. **Click "Load Data & Generate Embeddings"**
    
    4. **Start searching** for your perfect fashion vibe!
    """)
    
    # Show sample data structure
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
    Built with ❤️ using Streamlit & Sentence Transformers
</div>
""", unsafe_allow_html=True)