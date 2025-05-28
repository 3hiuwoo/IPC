from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_neo4j import Neo4jChatMessageHistory

from graph import graph
from llm import llm
from utils import get_session_id
from tools.cypher import cypher_qa
from tools.vector import retrieve_disease_description

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "你是一位专业医学专家，只能回答与医疗、健康、疾病、药物、症状等相关的问题。"
         "请用准确、详细且通俗易懂的语言解答用户问题，必要时可举例说明。"
         "请不要编造信息，无法确定时请直接说明。"
         "如遇紧急或严重情况，请建议用户及时就医。"),
        ("human", "{input}"),
    ]
)

consult_chat = chat_prompt | llm | StrOutputParser()

# Create a set of tools
tools = [
    Tool.from_function(
        name="通用对话",
        description="用于处理无法通过知识图谱检索到答案的医疗相关问题，提供专业医学建议。",
        func=consult_chat.invoke,
    ),
    Tool.from_function(
        name="医学描述语义检索",  
        description="用于根据用户问题，检索疾病等节点的详细描述、病因、治疗方法等长文本内容，适合‘什么是某疾病’等问题。",
        func=retrieve_disease_description, 
    ),
    Tool.from_function(
        name="医疗信息查询",
        description="基于医疗知识图谱，使用Cypher语句检索疾病、症状、药物等结构化信息，适合‘某疾病有哪些症状’、‘某症状可能是什么病’等问题。",
        func=cypher_qa
    )
]

# Create chat history callback
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Create the agent
agent_prompt = PromptTemplate.from_template("""
你是一位医学专家，专注于为用户提供权威、详细的医疗健康信息。

请尽可能详细、准确地回答用户的问题，只能基于提供的上下文和工具返回答案，不要凭空编造。

请不要回答与医疗、健康、疾病、药物、症状等无关的问题。

注意事项:
1.请注意，疾病、症状、药物等类型之间都是多对多关系，请在回答时尽量完整列出所有相关内容。
2.如获取信息不全或需进一步确认，可主动向用户提问以获取更多细节。
3.如果始终无法通过知识图谱或文本检索等工具无法获得有效答案，请直接告知用户“暂无相关知识”或“建议咨询专业医生”，不要编造信息。
4.由于数据清洗不彻底，有时候检索得到的数据可能出现错别字、冗余等情况，请在回答时尽量修正这些问题，确保回答的专业性和准确性。

工具列表如下：

{tools}

如需使用工具，请使用以下格式：

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

如果你已经可以直接回答，或不需要使用工具，请使用以下格式：

```
Thought: Do I need to use a tool? No
Final Answer: [你的回复]
```

开始！

历史对话记录:
{chat_history}

用户新问题: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
    )
chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Create a handler to call the agent
def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}})

    return response['output']