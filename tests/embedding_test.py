from sentence_transformers import SentenceTransformer, util

name = "voyageai/voyage-4-nano"
print("Embedding Model:", name)
model = SentenceTransformer(
    name,
    trust_remote_code=True,
    #model_kwargs={"default_task": "clustering"},
)

docs = [
    "Einstein went to columbia university",
]
query = "education"

doc_emb = model.encode(docs, convert_to_tensor=True)
query_emb = model.encode(query, convert_to_tensor=True)

scores = util.cos_sim(query_emb, doc_emb)[0]

for doc, score in zip(docs, scores):
    print(f"{score.item():.4f}  {doc}")