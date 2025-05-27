import streamlit as st
from llm import llm, embeddings
from graph import graph
from langchain_neo4j import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# Create the Neo4jVector
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="diseaseDescriptions",           
    node_label="Disease",                       
    text_node_property="desc",                  
    embedding_node_property="descEmbedding",    
    retrieval_query="""
RETURN
    node.desc AS text,
    {
        name: node.name,
        symptoms: [ (node)-[:has_symptom]->(symptom) | symptom.name ],
        recommend_eat: [ (node)-[:recommend_eat]->(recipe) | recipe.name ],
        recommend_drug: [ (node)-[:recommend_drug]->(drug) | drug.name ],
        cause: node.cause,
        cure_way: node.cure_way,
    } AS metadata
"""
)

# Create the retriever
retriever = neo4jvector.as_retriever()

# Create the prompt
instructions = (
    "请结合给定的医学背景知识（Context）用中文详细回答用户的问题。"
    "如果无法从Context中获得答案，请直接回复“我不知道”。"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instructions),
        ("human", "{input}"),
    ]
)

# Create the chain 
question_answer_chain = create_stuff_documents_chain(llm, prompt)
medical_retriever = create_retrieval_chain(
    retriever, 
    question_answer_chain
)

# Create a function to call the chain
def retrieve_disease_description(input):
    return medical_retriever.invoke({"input": input})