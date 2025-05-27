import streamlit as st

from agent import generate_response
from utils import write_message

# Page Config
st.set_page_config("智能问诊助手", page_icon=":male-doctor:")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是您的智能问诊助手，请问有什么可以帮助您的？"},
    ]

# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if question := st.chat_input("输入你的症状"):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    with st.spinner('思考中...'):
        response = generate_response(message)
        write_message('assistant', response)
