import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load the API Key securely from the .env file
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    print("Error: GROQ_API_KEY not found. Please check your .env file.")
    exit()

# 2. Re-load the Vector Database
print("Loading vector database...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Set up the retriever to fetch the top 3 most relevant chunks based on a question
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# 3. Initialize the LLM (Using Llama-3 via Groq for high-speed inference)
print("Initializing LLM...")
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# 4. Create the Prompt Template
# This explicitly instructs the AI to base its answer ONLY on the retrieved text
system_prompt = (
    "You are an intelligent assistant. Use the following pieces of retrieved context "
    "to answer the question. If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Build the RAG Chain
# This chain links the document retriever to the LLM prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Ask a Question!
# Change this string to a question that can be answered by your sample_data.txt
question = "What is the main topic of the text?" 

print(f"\nQuestion: {question}")
print("Thinking...")

# Invoke the chain
response = rag_chain.invoke({"input": question})

print(f"\nAnswer: {response['answer']}")