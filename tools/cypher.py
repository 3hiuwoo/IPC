import streamlit as st
from langchain.prompts.prompt import PromptTemplate
from langchain_neo4j import GraphCypherQAChain

from graph import graph
from llm import llm

# Create the Cypher QA chain
CYPHER_GENERATION_TEMPLATE = """
你是一位专业的Neo4j医疗知识图谱专家，请将用户的医疗健康相关问题转化为Cypher查询语句，用于查询疾病、症状、药物、检查等信息。

注意事项：
1. 所有节点/实体类型、关系类型、属性名称均为英文（如 Disease, Symptom, has_symptom, desc 等），请严格按照schema中的英文名称生成。
2. 所有属性的内容为中文（如疾病描述、症状名称等）。
3. 如有疑问，请参考下方schema中的中英文对照示例。

示例：
- 节点类型: Disease（疾病）, Symptom（症状）等
- 关系类型: has_symptom（有症状）, recommend_drug（推荐药物）等
- 属性名称: name（名称）, desc（描述）等
- 属性内容: "流感", "发热"等

请严格根据下方提供的知识图谱schema生成查询语句，只能使用schema中出现的节点类型、关系类型和属性。

不要返回整个节点或嵌入属性。

以下是一些Cypher查询语句示例:

1.查询某种症状对应的所有可能疾病:
```
MATCH (d:Disease)-[:has_symptom]->(s:Symptom {{name: "症状名称"}})
RETURN s.name AS symptom
```

2.查询得了某种疾病后不能吃的食物:
```
MATCH (d:Disease {{name: "疾病名称"}})-[:no_eat]->(r:Food)
RETURN r.name AS no_eat_food
```

3.查询得了某种疾病后推荐吃的菜肴：
```
MATCH (d:Disease {{name: "疾病名称"}})-[:recommend_eat]->(r:Recipe)
RETURN r.name AS recommend_eat_recipe
```

4.查询某种疾病的治疗方法：
```
MATCH (d:Disease {{name: "疾病名称"}})
RETURN d.cure_way AS cure_way
```

Schema:
{schema}

用户问题:
{question}

Cypher查询语句:
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)


cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True
)