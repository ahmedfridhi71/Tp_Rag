import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


# ============================================================
# 1. Configuration
# ============================================================

load_dotenv()

COLLECTION_NAME = "invoices"
QDRANT_PATH = "./qdrant_db"


# ============================================================
# 2. Embeddings
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 3. Connexion Qdrant
# ============================================================

print("Connecting to Qdrant...")

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    path=QDRANT_PATH,
)


# ============================================================
# 4. LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)


# ============================================================
# 5. Question
# ============================================================

question = input("\nYour question: ")


# ============================================================
# 6. Recherche
# ============================================================

documents = []

# Chercher un numéro de facture dans la question
numbers = re.findall(r"\b\d{3,6}\b", question)


if numbers:

    invoice_code = numbers[0]

    print(f"\nSearching invoice: {invoice_code}")

    # Accès au client Qdrant déjà utilisé par vector_store
    client = vector_store.client

    # Recherche exacte dans les payloads
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )

    for point in points:

        payload = point.payload or {}

        metadata = payload.get("metadata", {})

        if str(metadata.get("invoice_code")) == str(invoice_code):

            documents.append(
                {
                    "metadata": metadata,
                    "text": payload.get("page_content", ""),
                }
            )


# ============================================================
# 7. Si aucune facture exacte n'est trouvée
# ============================================================

if not documents:

    print("\nExact invoice not found.")

    print("Using semantic search...")

    semantic_documents = vector_store.similarity_search(
        question,
        k=5
    )

    for doc in semantic_documents:

        documents.append(
            {
                "metadata": doc.metadata,
                "text": doc.page_content,
            }
        )


# ============================================================
# 8. Afficher les documents trouvés
# ============================================================

print(f"\nDocuments found: {len(documents)}")

for i, doc in enumerate(documents, start=1):

    metadata = doc["metadata"]

    print(
        f"{i}. Invoice: {metadata.get('invoice_code')} | "
        f"Total: {metadata.get('total')}"
    )


# ============================================================
# 9. Construire le contexte
# ============================================================

context_parts = []

for doc in documents:

    metadata = doc["metadata"]

    context_parts.append(
        f"""
Invoice code: {metadata.get('invoice_code')}
Bill to: {metadata.get('bill_to')}
Ship to: {metadata.get('ship_to')}
Total: {metadata.get('total')}
Shipping: {metadata.get('shipping')}
Discount: {metadata.get('discount')}
Amount: {metadata.get('amount')}

Invoice text:
{doc["text"]}
"""
    )


context = "\n\n".join(context_parts)


# ============================================================
# 10. LLM
# ============================================================

prompt = f"""
You are an assistant specialized in invoices.

Answer the user's question using ONLY the information below.

Do not invent information.

Invoice information:

{context}

User question:

{question}

Give a short and precise answer.
"""


response = llm.invoke(prompt)


# ============================================================
# 11. Answer
# ============================================================

print("\n================ ANSWER ================\n")

print(response.content)