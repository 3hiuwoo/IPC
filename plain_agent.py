from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from graph import graph
from llm import llm

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "你是一位专业医学专家，只能回答与医疗、健康、疾病、药物、症状等相关的问题。"
         "请用准确、详细且通俗易懂的语言解答用户问题，必要时可举例说明。"
         "请不要编造信息，无法确定时请直接说明。"
         "如遇紧急或严重情况，请建议用户及时就医。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

consult_chat = chat_prompt | llm | StrOutputParser()

memory = ChatMessageHistory()

def get_memory(session_id):
    return memory

chat_with_message_history = RunnableWithMessageHistory(
    consult_chat,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# Create a handler to call the agent
def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_with_message_history.invoke(
        {"input": user_input},
        {"configurable": {"session_id": "none"}})

    return response