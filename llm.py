import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Create the LLM
llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    base_url=st.secrets["OPENAI_BASE_URL"],
    model=st.secrets["OPENAI_MODEL"],
)

# Create the Embedding model
embeddings = OpenAIEmbeddings(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    base_url=st.secrets["OPENAI_BASE_URL"]
)