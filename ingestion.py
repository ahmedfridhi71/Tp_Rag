import os
import time

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from metadata_extractor import read_pdf, extract_metadata


# ============================================================
# 1. Parametres
# ============================================================

INVOICES_DIR = os.path.join("1000+ PDF_Invoice_Folder", "1000+ PDF_Invoice_Folder")
COLLECTION_NAME = "invoices"
QDRANT_PATH = "qdrant_db"
VECTOR_SIZE = 384   # taille des vecteurs du modele all-MiniLM-L6-v2

# Mettre un nombre (ex: 10) pour tester sur peu de PDF, None pour tout traiter
LIMIT = None


# ============================================================
# 2. Embeddings (locaux, gratuits)
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 3. Creer / ouvrir la collection "invoices" dans Qdrant
# ============================================================

client = QdrantClient(path=QDRANT_PATH)

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' creee")
else:
    print(f"Collection '{COLLECTION_NAME}' existe deja")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)


# ============================================================
# 4. Lister les PDF du dossier
# ============================================================

pdf_files = sorted(
    f for f in os.listdir(INVOICES_DIR) if f.lower().endswith(".pdf")
)

if LIMIT:
    pdf_files = pdf_files[:LIMIT]

print(f"\n{len(pdf_files)} PDF a traiter dans '{INVOICES_DIR}'\n")


# ============================================================
# 5. Ingestion : extraction + metadonnees + stockage
# ============================================================

success = 0
skipped = []
warnings = []
errors = []

for i, filename in enumerate(pdf_files, start=1):
    path = os.path.join(INVOICES_DIR, filename)

    try:
        invoice_text = read_pdf(path)

        if not invoice_text.strip():
            raise ValueError("PDF sans texte")

        meta = extract_metadata(invoice_text)

        # Facture vide (pas de numero, pas de client) -> on ignore
        if not meta.invoice_code or not meta.bill_to:
            skipped.append(filename)
            print(f"[{i}/{len(pdf_files)}] IGNOREE {filename} (facture vide)")
            continue

        total = meta.total if meta.total is not None else 0.0
        amount = meta.amount if meta.amount is not None else 0.0

        # Verification : amount - discount + shipping = total
        expected = round(amount - meta.discount + meta.shipping, 2)
        if abs(expected - total) > 0.05:
            warnings.append(
                (filename, f"amount-discount+shipping={expected} mais total={total}")
            )

        metadata = {
            "source": filename,
            "invoice_code": meta.invoice_code,
            "bill_to": meta.bill_to,
            "ship_to": meta.ship_to or "",
            "total": total,
            "shipping": meta.shipping,
            "discount": meta.discount,
            "amount": amount,
        }

        doc = Document(page_content=invoice_text, metadata=metadata)

        # id numerique stable derive du code facture -> pas de doublon si on relance
        point_id = int(meta.invoice_code) if meta.invoice_code.isdigit() else i
        vectorstore.add_documents([doc], ids=[point_id])

        success += 1
        print(f"[{i}/{len(pdf_files)}] OK      {filename} -> {meta.invoice_code}")

    except Exception as e:
        errors.append((filename, str(e)))
        print(f"[{i}/{len(pdf_files)}] ERREUR  {filename} : {e}")

    time.sleep(0.5)  # evite la limite de debit de Groq


# ============================================================
# 6. Resume
# ============================================================

print("\n================ RESUME ================\n")
print("Reussies  :", success)
print("Ignorees  :", len(skipped))
print("Warnings  :", len(warnings))
print("Echouees  :", len(errors))

for name in skipped:
    print(" - ignoree :", name)

for name, msg in warnings:
    print(" - a verifier :", name, "->", msg)

for name, err in errors:
    print(" - erreur :", name, ":", err)

print("\nPoints dans la collection :", client.count(COLLECTION_NAME).count)
print("\n=== ETAPE 3 OK ===")