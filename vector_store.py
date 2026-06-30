from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Read and Split the Text (Reusing your chunker logic)
file_path = "sample_data.txt"
with open(file_path, "r", encoding="utf-8") as file:
    raw_text = file.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, length_function=len
)
chunks = text_splitter.split_text(raw_text)

print(f"Loaded {len(chunks)} chunks.")

# 2. Initialize the Embedding Model
# This will download a small (~80MB) open-source embedding model the first time you run it.
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Create and Persist the Vector Store
# This creates a local folder called "chroma_db" to store the embeddings.
persist_directory = "./chroma_db"
print("Generating embeddings and saving to ChromaDB...")

vector_db = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    persist_directory=persist_directory
)

vector_db.persist()
print("Vector database created successfully! Ready for querying.")