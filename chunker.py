from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Read the source document
file_path = "sample_data.txt"
try:
    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()
except FileNotFoundError:
    print(f"Error: Please create a file named '{file_path}' and add some text.")
    exit()

# 2. Configure the Text Splitter
# chunk_size: Max characters per chunk
# chunk_overlap: Characters shared between consecutive chunks to preserve context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       
    chunk_overlap=50,     
    length_function=len,
    separators=["\n\n", "\n", " ", ""] 
)

# 3. Execute the splitting
chunks = text_splitter.split_text(raw_text)

# 4. Display the results
print(f"Original Text Length: {len(raw_text)} characters")
print(f"Total Chunks Created: {len(chunks)}\n")

# Print the first 3 chunks to verify the overlap and structure
for i, chunk in enumerate(chunks[:3]): 
    print(f"--- Chunk {i+1} (Length: {len(chunk)}) ---")
    print(chunk)
    print("-" * 50 + "\n")