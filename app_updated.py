import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import time
import json
import requests
from sentence_transformers import SentenceTransformer, util
from serpapi import GoogleSearch
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# CONFIGURATION
# ============================================================================

# Page config
st.set_page_config(
    page_title="Vibe Matcher 🎨",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Keys - Add to Streamlit secrets or sidebar input
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")

# Model configuration
EMBEDDING_MODEL = "intfloat/e5-base-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ============================================================================
# SAMPLE FASHION DATA
# ============================================================================

FASHION_DATA = [
    {
        "name": "Boho Maxi Dress",
        "description": "Flowy maxi dress with earthy tones, perfect for festival vibes. Features intricate embroidery and relaxed fit.",
        "vibes": ["boho", "festival", "relaxed", "earthy"],
        "price": "₹2,499",
        "category": "Dresses",
        "color": "Earth tones"
    },
    {
        "name": "Urban Street Jacket",
        "description": "Edgy black leather jacket with asymmetric zipper. Perfect for energetic urban chic street style.",
        "vibes": ["urban", "edgy", "street", "energetic"],
        "price": "₹4,999",
        "category": "Outerwear",
        "color": "Black"
    },
    {
        "name": "Cozy Oversized Sweater",
        "description": "Soft cashmere blend oversized sweater in cream. Ultimate comfort for cozy indoor vibes.",
        "vibes": ["cozy", "comfort", "minimalist", "soft"],
        "price": "₹3,299",
        "category": "Knitwear",
        "color": "Cream"
    },
    {
        "name": "Vintage Floral Midi Skirt",
        "description": "Romantic midi skirt with vintage floral print. Flowing fabric perfect for feminine cottagecore aesthetics.",
        "vibes": ["vintage", "romantic", "cottagecore", "feminine"],
        "price": "₹1,899",
        "category": "Skirts",
        "color": "Floral"
    },
    {
        "name": "Minimalist Linen Shirt",
        "description": "Clean-cut linen shirt in white. Breathable and sophisticated for minimalist modern styling.",
        "vibes": ["minimalist", "clean", "modern", "sophisticated"],
        "price": "₹2,199",
        "category": "Tops",
        "color": "White"
    },
    {
        "name": "Grunge Distressed Jeans",
        "description": "High-waisted distressed denim with raw hems. Rebellious grunge vibes with vintage wash.",
        "vibes": ["grunge", "rebellious", "vintage", "edgy"],
        "price": "₹2,799",
        "category": "Bottoms",
        "color": "Denim"
    },
    {
        "name": "Ethereal Silk Blouse",
        "description": "Delicate silk blouse in soft lavender. Ethereal and dreamy for romantic occasions.",
        "vibes": ["ethereal", "romantic", "dreamy", "delicate"],
        "price": "₹3,499",
        "category": "Tops",
        "color": "Lavender"
    },
    {
        "name": "Athletic Jogger Set",
        "description": "Sleek athleisure jogger set in black. Modern sporty vibes with moisture-wicking fabric.",
        "vibes": ["athletic", "sporty", "modern", "energetic"],
        "price": "₹2,999",
        "category": "Activewear",
        "color": "Black"
    },
    {
        "name": "Leather Biker Boots",
        "description": "Rugged leather boots with buckle details. Perfect for edgy rock and roll style with a rebellious attitude.",
        "vibes": ["edgy", "rock", "rebellious", "tough"],
        "price": "₹5,499",
        "category": "Footwear",
        "color": "Black"
    },
    {
        "name": "Preppy Pleated Skirt",
        "description": "Classic pleated tennis skirt in navy. Preppy academic vibes with vintage collegiate charm.",
        "vibes": ["preppy", "academic", "classic", "youthful"],
        "price": "₹1,699",
        "category": "Skirts",
        "color": "Navy"
    },
    {
        "name": "Tropical Print Shirt",
        "description": "Vibrant Hawaiian shirt with palm leaf print. Bold vacation vibes for carefree summer energy.",
        "vibes": ["tropical", "vacation", "bold", "carefree"],
        "price": "₹2,299",
        "category": "Tops",
        "color": "Multi"
    },
    {
        "name": "Monochrome Blazer",
        "description": "Structured black blazer with sharp tailoring. Professional power dressing for corporate sophistication.",
        "vibes": ["professional", "corporate", "sophisticated", "powerful"],
        "price": "₹6,499",
        "category": "Outerwear",
        "color": "Black"
    },
    {
        "name": "Bohemian Kimono",
        "description": "Flowing kimono with paisley print and fringe details. Free-spirited boho vibes for festival season.",
        "vibes": ["boho", "free-spirited", "festival", "artistic"],
        "price": "₹3,799",
        "category": "Outerwear",
        "color": "Mixed"
    },
    {
        "name": "Neon Crop Top",
        "description": "Electric neon crop top in lime green. High-energy rave vibes for bold party looks.",
        "vibes": ["bold", "energetic", "party", "rave"],
        "price": "₹1,299",
        "category": "Tops",
        "color": "Neon Green"
    },
    {
        "name": "Cottagecore Dress",
        "description": "Puff sleeve midi dress with delicate floral embroidery. Soft romantic cottagecore aesthetic for dreamy afternoons.",
        "vibes": ["cottagecore", "romantic", "soft", "dreamy"],
        "price": "₹3,999",
        "category": "Dresses",
        "color": "Pastel"
    },
    {
        "name": "Tech Wear Cargo Pants",
        "description": "Water-resistant cargo pants with multiple pockets. Futuristic techwear aesthetic with urban functionality.",
        "vibes": ["techwear", "futuristic", "urban", "functional"],
        "price": "₹4,299",
        "category": "Bottoms",
        "color": "Charcoal"
    },
    {
        "name": "Sunset Tie-Dye Hoodie",
        "description": "Hand-dyed hoodie with orange and pink gradient. Chill laid-back vibes with artistic hippie influence.",
        "vibes": ["hippie", "artistic", "chill", "laid-back"],
        "price": "₹2,599",
        "category": "Knitwear",
        "color": "Tie-Dye"
    },
    {
        "name": "Gothic Velvet Dress",
        "description": "Deep burgundy velvet dress with Victorian collar. Dark romantic gothic vibes for mysterious elegance.",
        "vibes": ["gothic", "dark", "romantic", "mysterious"],
        "price": "₹5,299",
        "category": "Dresses",
        "color": "Burgundy"
    },
    {
        "name": "Scandinavian Wool Sweater",
        "description": "Fair isle pattern sweater in Nordic style. Hygge-inspired cozy vibes with sustainable wool.",
        "vibes": ["hygge", "cozy", "nordic", "sustainable"],
        "price": "₹4,799",
        "category": "Knitwear",
        "color": "Gray/White"
    },
    {
        "name": "Y2K Mini Skirt",
        "description": "Low-rise denim mini skirt with rhinestone details. Nostalgic Y2K vibes with playful 2000s energy.",
        "vibes": ["y2k", "nostalgic", "playful", "trendy"],
        "price": "₹1,999",
        "category": "Skirts",
        "color": "Light Denim"
    }
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model"""
    return SentenceTransformer(EMBEDDING_MODEL)

# @st.cache_data
# def create_product_embeddings(_model, products_df):
#     """Create embeddings for all products"""
#     texts = [
#         f"{row['name']}: {row['description']} Vibes: {', '.join(row['vibes'])}"
#         for _, row in products_df.iterrows()
#     ]
#     embeddings = _model.encode(texts, convert_to_tensor=True)
#     return embeddings

@st.cache_data
def create_product_embeddings(_model, products_df):
    """Create normalized, context-enriched embeddings for all products."""
    texts = [
        f"passage: Product {row['name']}. Description: {row['description']}. "
        f"Category: {row['category']}. Color: {row['color']}. "
        f"Vibes: {', '.join(row['vibes'])}. Style keywords: fashion, outfit, clothing."
        for _, row in products_df.iterrows()
    ]
    embeddings = _model.encode(texts, convert_to_tensor=True, normalize_embeddings=True) # normalize for better cosine stability
    return embeddings

def call_groq_api(messages: List[Dict], api_key: str, temperature: float = 0.7) -> str:
    """Call GROQ API for text generation"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling GROQ API: {str(e)}"

def search_web(query: str, serpapi_key: str, sites: List[str] = ["amazon.in", "myntra.com"]) -> List[Dict]:
    """Search web for products using SERP API"""
    all_results = []
    
    for site in sites:
        try:
            search_query = f"{query} clothing or fashion site:{site}"
            search = GoogleSearch({
                "q": search_query,
                "api_key": serpapi_key,
                "num": 3
            })
            results = search.get_dict()
            
            if "organic_results" in results:
                for result in results["organic_results"][:3]:
                    all_results.append({
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "source": site
                    })
        except Exception as e:
            st.warning(f"Error searching {site}: {e}")
    
    return all_results

# def compute_similarity(query: str, model, product_embeddings, products_df, top_k: int = 3):
#     """Compute cosine similarity and return top-k matches"""
#     query_embedding = model.encode(query, convert_to_tensor=True)
#     cos_scores = util.cos_sim(query_embedding, product_embeddings)[0]
    
#     top_results = np.argsort(cos_scores.cpu().numpy())[-top_k:][::-1]
    
#     recommendations = []
#     for idx in top_results:
#         score = float(cos_scores[idx])
#         product = products_df.iloc[idx]
        
#         recommendations.append({
#             "name": product["name"],
#             "description": product["description"],
#             "vibes": product["vibes"],
#             "price": product["price"],
#             "category": product["category"],
#             "color": product["color"],
#             "similarity_score": score,
#             "match_quality": "Excellent" if score > 0.7 else "Good" if score > 0.5 else "Fair"
#         })
    
#     return recommendations

def compute_similarity(query: str, model, product_embeddings, products_df, top_k: int = 3):
    """Compute cosine similarity and return top-k matches"""
    query_text = f"query: {query}"
    query_embedding = model.encode(
        query_text,
        convert_to_tensor=True,
        normalize_embeddings=True     # 💡 normalization handled automatically
    )
    
    cos_scores = util.cos_sim(query_embedding, product_embeddings)[0]
    
    top_results = np.argsort(cos_scores.cpu().numpy())[-top_k:][::-1]
    recommendations = []
    for idx in top_results:
        score = float(cos_scores[idx])
        product = products_df.iloc[idx]
        
        recommendations.append({
            "name": product["name"],
            "description": product["description"],
            "vibes": product["vibes"],
            "price": product["price"],
            "category": product["category"],
            "color": product["color"],
            "similarity_score": score,
            "match_quality": "Excellent" if score > 0.7 else "Good" if score > 0.5 else "Fair"
        })
    return recommendations


def generate_response(query: str, rag_results: List[Dict], web_results: List[Dict], groq_api_key: str) -> str:
    """Generate expressive fashion stylist response using GROQ API."""

    # Format RAG results (curated items)
    rag_text = "\n".join([
        f"- {r['name']} ({r['price']}): {r['description'][:120]} "
        f"[Match Quality: {r['match_quality']}, Score: {r['similarity_score']:.2f}]"
        for r in rag_results
    ])

    # Format web search results (if any)
    web_text = "\n".join([
        f"- {w['title']} ({w['source']})"
        for w in web_results[:5]
    ]) if web_results else "No external shopping links found."

    # --- SYSTEM PROMPT (defines assistant persona) ---
    system_prompt = """
    You are "Vibe Matcher" — an expert AI Fashion Stylist.
    You understand aesthetics, color harmony, body-fit preferences, and trending outfit combinations.

    Your goal:
    - Understand the user's vibe or occasion (e.g., "boho festival look", "romantic date night").
    - Curate and describe 2–3 outfit recommendations that best match their style.
    - Use the provided product results (from internal catalog) as your foundation.
    - Optionally, mention helpful online shopping links from trusted sources.
    - Write in a warm, creative stylist tone — friendly, confident, and fashion-savvy.

    Your response must:
    1. Greet the user naturally and acknowledge their vibe or style goal.
    2. Highlight the top matching pieces with short stylist reasoning.
    3. Suggest how to pair or accessorize the outfits (optional).
    4. Reference external links if relevant.
    Keep it around 2-3 short paragraphs.
    Avoid technical language or raw data.
    """

    # --- USER PROMPT (injects contextual data) ---
    user_prompt = f"""
    User Query: {query}

    Top Retrieved Fashion Items:
    {rag_text}

    Additional Shopping Links:
    {web_text}

    Now, write a short, stylish recommendation that:
    - Interprets the user's vibe in context.
    - Suggests the most fitting pieces from above.
    - Explains the reasoning creatively (color, silhouette, vibe).
    - Keeps tone conversational and stylistic.
    """

    # Construct messages for the model
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    # Call GROQ API
    return call_groq_api(messages, groq_api_key)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "products_df" not in st.session_state:
    st.session_state.products_df = pd.DataFrame(FASHION_DATA)

if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = []

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Keys input
    st.subheader("API Keys")
    groq_key = st.text_input("GROQ API Key", value=GROQ_API_KEY, type="password")
    serp_key = st.text_input("SERP API Key", value=SERPAPI_KEY, type="password")
    
    st.divider()
    
    # Settings
    st.subheader("Settings")
    top_k = st.slider("Top K Recommendations", 1, 5, 3)
    enable_web_search = st.checkbox("Enable Web Search", value=True)
    search_sites = st.multiselect(
        "Search Sites",
        ["amazon.in", "myntra.com", "ajio.com"],
        default=["amazon.in", "myntra.com"]
    )
    
    st.divider()
    
    # Stats
    st.subheader("📊 Stats")
    st.metric("Total Queries", len(st.session_state.conversation_history))
    st.metric("Products in DB", len(st.session_state.products_df))
    
    if st.button("🗑️ Clear History"):
        st.session_state.conversation_history = []
        st.session_state.metrics_history = []
        st.rerun()

# ============================================================================
# MAIN APP
# ============================================================================

st.title("🎨 Vibe Matcher")
st.markdown("### Find fashion that matches your vibe using AI-powered recommendations")

# Check API keys
if not groq_key or not serp_key:
    st.warning("⚠️ Please enter your API keys in the sidebar to get started!")
    st.info("""
    **Get your API keys:**
    - GROQ: https://console.groq.com
    - SERP API: https://serpapi.com
    """)
    st.stop()

# Load models
with st.spinner("Loading AI models..."):
    embedding_model = load_embedding_model()
    product_embeddings = create_product_embeddings(embedding_model, st.session_state.products_df)

# Query input
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input(
        "Enter your vibe:",
        placeholder="e.g., energetic urban chic, cozy winter vibes, romantic date night...",
        label_visibility="collapsed"
    )
with col2:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

# Quick suggestions
st.markdown("**Quick suggestions:**")
suggestions = ["energetic urban chic", "cozy comfortable vibes", "romantic ethereal outfit", "grunge street style"]
cols = st.columns(4)
for idx, suggestion in enumerate(suggestions):
    if cols[idx].button(suggestion, key=f"sug_{idx}"):
        user_query = suggestion
        search_button = True

# Process query
if search_button and user_query:
    with st.spinner("✨ Finding your perfect match..."):
        start_time = time.time()
        
        # Step 1: RAG recommendations
        rag_start = time.time()
        rag_results = compute_similarity(
            user_query, 
            embedding_model, 
            product_embeddings, 
            st.session_state.products_df, 
            top_k=top_k
        )
        rag_latency = time.time() - rag_start
        
        # Step 2: Web search
        web_results = []
        web_latency = 0
        if enable_web_search:
            web_start = time.time()
            web_results = search_web(user_query, serp_key, search_sites)
            web_latency = time.time() - web_start
        
        # Step 3: Generate response
        gen_start = time.time()
        response = generate_response(user_query, rag_results, web_results, groq_key)
        gen_latency = time.time() - gen_start
        
        total_latency = time.time() - start_time
        
        # Store in history
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": user_query,
            "rag_results": rag_results,
            "web_results": web_results,
            "response": response,
            "metrics": {
                "rag_latency": rag_latency,
                "web_latency": web_latency,
                "gen_latency": gen_latency,
                "total_latency": total_latency
            }
        }
        st.session_state.conversation_history.append(conversation_entry)
        st.session_state.metrics_history.append({
            "query": user_query,
            "total_latency": total_latency,
            "timestamp": datetime.now()
        })

# Display results
if st.session_state.conversation_history:
    latest = st.session_state.conversation_history[-1]
    
    # AI Response
    st.markdown("---")
    st.markdown("### 💬 AI Stylist Response")
    st.markdown(latest["response"])
    
    # Tabs for different result types
    tab1, tab2, tab3, tab4 = st.tabs(["✨ Curated Picks", "🔗 Shopping Links", "📊 Analytics", "💾 History"])
    
    with tab1:
        st.markdown("### Top Curated Recommendations")
        for idx, rec in enumerate(latest["rag_results"], 1):
            with st.expander(f"{idx}. {rec['name']} - {rec['price']}", expanded=(idx==1)):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Description:** {rec['description']}")
                    st.markdown(f"**Vibes:** {', '.join(rec['vibes'])}")
                    st.markdown(f"**Category:** {rec['category']} | **Color:** {rec['color']}")
                with col2:
                    score_color = "🟢" if rec['match_quality'] == "Excellent" else "🟡" if rec['match_quality'] == "Good" else "🟠"
                    st.metric("Match Quality", f"{score_color} {rec['match_quality']}")
                    st.metric("Similarity Score", f"{rec['similarity_score']:.3f}")
    
    with tab2:
        st.markdown("### Shopping Links")
        if latest["web_results"]:
            for idx, link in enumerate(latest["web_results"], 1):
                st.markdown(f"""
                **{idx}. {link['title']}**  
                🏪 Source: {link['source']}  
                📝 {link['snippet'][:150]}...  
                🔗 [Visit Link]({link['link']})
                """)
                st.divider()
        else:
            st.info("No web links found. Try enabling web search in settings.")
    
    with tab3:
        st.markdown("### Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RAG Search", f"{latest['metrics']['rag_latency']:.3f}s")
        col2.metric("Web Search", f"{latest['metrics']['web_latency']:.3f}s")
        col3.metric("AI Generation", f"{latest['metrics']['gen_latency']:.3f}s")
        col4.metric("Total Time", f"{latest['metrics']['total_latency']:.3f}s")
        
        # Latency chart
        if len(st.session_state.metrics_history) > 1:
            st.markdown("#### Latency Trend")
            df_metrics = pd.DataFrame(st.session_state.metrics_history)
            fig = px.line(df_metrics, x=df_metrics.index, y='total_latency', 
                         labels={'total_latency': 'Latency (s)', 'index': 'Query #'},
                         title='Query Latency Over Time')
            st.plotly_chart(fig, use_container_width=True)
        
        # Similarity distribution
        st.markdown("#### Similarity Score Distribution")
        scores = [r['similarity_score'] for r in latest['rag_results']]
        names = [r['name'] for r in latest['rag_results']]
        fig = go.Figure(data=[go.Bar(x=names, y=scores, marker_color='lightblue')])
        fig.update_layout(yaxis_title='Similarity Score', xaxis_title='Product')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Conversation History")
        if st.session_state.conversation_history:
            for idx, entry in enumerate(reversed(st.session_state.conversation_history), 1):
                with st.expander(f"Query {len(st.session_state.conversation_history) - idx + 1}: {entry['query']}", expanded=(idx==1)):
                    st.markdown(f"**Timestamp:** {entry['timestamp']}")
                    st.markdown(f"**Response:** {entry['response'][:200]}...")
                    st.markdown(f"**Latency:** {entry['metrics']['total_latency']:.3f}s")
        else:
            st.info("No conversation history yet.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with ❤️ using Sentence Transformers, GROQ API & SERP API | 
    <a href='https://github.com' target='_blank'>GitHub</a>
</div>
""", unsafe_allow_html=True)