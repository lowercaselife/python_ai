import chromadb

client = chromadb.Client()

collection = client.create_collection("news_atricles")

collection.add(
    ids=["id1", "id2", "id3", "id4"],
    documents=[
        "Apple is leading in a smart phone game with iPhone sales up by 35%",
        "Tesla booked a minor profit of 1 billion $ in Q2",
        "Apples are high in fiber, vitamin C, and various antioxidants",
        "SpaceX got NASA contract worth 10 billion $",
    ]
)


data = collection.get(
    include = ["documents", "metadatas", "embeddings"]
)


for i, emb in enumerate(data["embeddings"]):
    print(f"\nDocument {i +1}")
    print("Text:", data["documents"][i])
    print("Embedding length:", len(emb))
    print("First 10 values of the embedding:", emb[:10])


result = collection.query(
    query_texts=["What is the profit of Tesla in Q2?"],
    n_results=2
)

print(result)

