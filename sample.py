import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 1. Load and chunk PDF
def extract_chunks_from_pdf(path, chunk_size=500):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

pdf_path = "pdfs/21BPS1494.pdf"
chunks = extract_chunks_from_pdf(pdf_path)

# 2. Embed chunks
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)
embeddings = embedder.encode(chunks)

# 3. Create FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# 4. Load Flan-T5
qa_model = pipeline("text2text-generation", model="google/flan-t5-xl", max_length=256)

# 5. Ask question
def get_answer(question):
    question_vec = embedder.encode([question])
    D, I = index.search(question_vec, k=3)
    # context = "\n".join([chunks[i] for i in I[0]])
    context = chunks[I[0][0]][:1000]
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    return qa_model(prompt)[0]['generated_text']

# CLI
while True:
    q = input("Ask a question (or type 'exit'): ")
    if q.lower() == "exit":
        break
    print("Answer:", get_answer(q))
