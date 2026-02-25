import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Entity Resolution Debugger", layout="wide")
st.title("Neutral Agent - Pipeline Trace Debugger")

# 1. Load the Traces
@st.cache_data
def load_data():
    try:
        with open("pipeline_traces.jsonl", "r") as f:
            data = [json.loads(line) for line in f]
        return pd.DataFrame(data)
    except FileNotFoundError:
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("No traces found. Run some queries in your API first!")
    st.stop()

# Flatten a few columns for the high-level table view
df['status'] = df['outcome'].apply(lambda x: x.get('status'))
df['latency'] = df['outcome'].apply(lambda x: x.get('latency_ms'))

# 2. Sidebar Filters
st.sidebar.header("Filters")
filter_status = st.sidebar.selectbox("Outcome Status", ["All", "success", "not_found", "error"])
if filter_status != "All":
    df = df[df['status'] == filter_status]

# 3. The Main Table
st.subheader("Recent Queries")
# Show a simplified view to the user
display_df = df[['timestamp', 'query', 'status', 'latency']].sort_values('timestamp', ascending=False)
selected_indices = st.dataframe(display_df, on_select="rerun", selection_mode="single-row")

# 4. The Deep Dive (JSON Trace)
if selected_indices and len(selected_indices.selection.rows) > 0:
    selected_row_idx = selected_indices.selection.rows[0]
    # Get the original complex JSON row
    selected_trace = df.iloc[selected_row_idx].to_dict()
    
    st.divider()
    st.subheader(f"Trace Deep Dive: \n Query: '{selected_trace['query']}'")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**LLM Reasoning:**")
        st.info(selected_trace['llm_judge'].get('reasoning'))
        
        st.markdown("**Top Retrieval Candidates:**")
        st.write(selected_trace['retrieval']['top_candidates'])
        
    with col2:
        st.markdown("**Raw JSON Trace:**")
        # st.json gives a beautiful, collapsible JSON tree!
        st.json(selected_trace)