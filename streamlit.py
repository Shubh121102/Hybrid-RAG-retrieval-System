import streamlit as st
import requests
import json

# ====================== CONFIG ======================
API_URL = "http://127.0.0.1:8030/rag/stream_answer"   # Change if needed

st.set_page_config(
    page_title="RAG Chat",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chat Assistant")
st.markdown("Ask a question about the Nike 10-K document")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API Endpoint", value=API_URL)
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Make sure FastAPI is running")
    st.markdown("2. Type your question")
    st.markdown("3. Click **Ask**")

# ====================== CHAT INPUT ======================
question = st.text_input("Your Question", placeholder="e.g. How many distribution centres in the US?")

ask_button = st.button("Ask", type="primary")

# ====================== STREAMING RESPONSE ======================
if ask_button and question.strip():
    answer_placeholder = st.empty()
    full_answer = ""

    with st.spinner("Thinking..."):
        try:
            with requests.post(
                api_url,
                json={"question": question},
                stream=True,
                timeout=120
            ) as response:

                if response.status_code != 200:
                    st.error(f"Error: {response.status_code} - {response.text}")
                else:
                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                data = json.loads(line)

                                if data.get("type") == "answer_chunk":
                                    chunk = data.get("content", "")
                                    full_answer += chunk
                                    # Update the UI in real-time
                                    answer_placeholder.markdown(full_answer + "▌")

                                elif data.get("type") == "final_message":
                                    answer_placeholder.markdown(full_answer)
                                    st.success("Streaming completed!")

                            except json.JSONDecodeError:
                                continue

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Is FastAPI running?")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

elif ask_button and not question.strip():
    st.warning("Please enter a question.")