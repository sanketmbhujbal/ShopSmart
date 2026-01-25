import streamlit as st
import requests

st.set_page_config(page_title="ShopSmart AI", page_icon="🛍️", layout="wide")

st.markdown("""
<style>
    .product-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        height: 380px;
        display: flex;
        flex-direction: column;
        transition: transform 0.2s;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .image-container {
        height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        margin-bottom: 10px;
    }
    .title-box {
        height: 50px;
        overflow: hidden;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 5px;
        line-height: 1.3em;
        color: #333;
    }
    .match-tag {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛍️ ShopSmart AI")
st.markdown("##### Intelligent Product Search")

query = st.text_input("Search", placeholder="Search ShopSmart", label_visibility="collapsed")
search_btn = st.button("Search", type="primary")

if search_btn or query:
    if query:
        try:
            response = requests.post("http://127.0.0.1:8000/search", json={"query": query, "top_k": 8})
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                if not results:
                    st.info("We couldn't find anything for that. Try something else.")
                else:
                    cols = st.columns(4)
                    for idx, item in enumerate(results):
                        with cols[idx % 4]:
                            # Image Logic
                            img_url = item['image'] if item['image'] else "https://via.placeholder.com/300"
                            
                            # Score Display (Clean percentage)
                            match_score = int(item['relevance_score'] * 100)
                            
                            st.markdown(f"""
                            <div class="product-card">
                                <div class="image-container">
                                    <img src="{img_url}" onerror="this.src='https://via.placeholder.com/300'" style="max-height: 100%; max-width: 100%;">
                                </div>
                                <div class="title-box">
                                    {item['title'][:60]}...
                                </div>
                                <div style="margin-top: auto;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                        <span style="font-size: 1.1rem; font-weight: bold; color: #b71c1c;">₹{item['price']}</span>
                                        <span style="font-size: 0.9rem;">⭐ {item['rating']}</span>
                                    </div>
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <span class="match-tag">{match_score}% Match</span>
                                        <span style="font-size: 0.75rem; color: gray;">{item['category'][:15]}</span>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.error("Service is temporarily unavailable.")
        except Exception:
            st.error("Connection Error. Please ensure the backend is running.")