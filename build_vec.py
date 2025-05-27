from llm import embeddings
from graph import graph
from tqdm import tqdm 

def main():
    results = graph.query("""
                        MATCH (d:Disease)
                        WHERE d.desc IS NOT NULL AND d.desc <> ''
                        RETURN d.name AS name, d.desc AS desc
                        """)

    for record in tqdm(results, desc="Vectorizing Disease Descriptions"):
        name = record["name"]
        desc = record["desc"]
        if desc:
            try:
                vector = embeddings.embed_query(desc)
                graph.query(
                    "MATCH (d:Disease {name: $name}) SET d.descEmbedding = $embedding",
                    params={"name": name, "embedding": vector}
                )
            except Exception as e:
                print(f"Error occur when creating embbeding: {e}")

    try:
        graph.query("""
        CREATE VECTOR INDEX diseaseDescriptions IF NOT EXISTS
        FOR (d:Disease)
        ON (d.descEmbedding)
        OPTIONS {indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
        }};
        """)
        print("Vector index created successfully.")
    except Exception as e:
        print(f"Error occur when creating vector index: {e}")
        
if __name__ == "__main__":
    main()