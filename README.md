# Medical Agent for Intelligent Pre-Consultation

- An Chinese medical chatbot that leverages a medical knowledge graph and Retrieval-Augmented Generation (RAG) to provide **Intelligent Pre-Consultation (IPC)** services.
- Using [Neo4j](https://neo4j.com/), it integrates both structured and unstructured information, enabling accurate and comprehensive answers to user health queries.
- Note that it is easy to transfer this project into English, as long as the English data is available.

## Running

Create a virtual environment and run:

```bash
pip install -r requirements.txt
```

Navigate to [.streamlit](.streamlit) and create a new file secrets.toml following [.streamlit/example.toml](.streamlit/example.toml) with your own information with OpenAI and Neo4j.

Build the knowledge graph:

```bash
python build_kg.py
```

(Optional) Build the vector base:

```bash
python build_vec.py
```

Run:

```bash
streamlit run bot.py
```

## Reference

1. [Neo4j GraphAcademy Course #1](https://graphacademy.neo4j.com/courses/llm-chatbot-python/)
2. [Neo4j GraphAcademy Course #2](https://graphacademy.neo4j.com/courses/llm-fundamentals/)

## Future work

- Data Cleaning
