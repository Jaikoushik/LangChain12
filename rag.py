from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

models = ChatGroq(model='llama-3.3-70b-versatile')

print("Enter the PDF Path")
pdf_path = input()

loader = PyPDFLoader(pdf_path)
a = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=20,
    length_function=len
)

splitted_text = text_splitter.split_documents(a)
#for idx, chunk in enumerate(splitted_text):
#    print('Chunk', {idx+1} ,':', repr(chunk.page_content))

query = input("Enter your query: ")

vector_store = Chroma(
    embedding_function=HuggingFaceEmbeddings(),
    persist_directory='chroma',
    collection_name='first'
)

vector_store.add_documents(splitted_text)

res = vector_store.similarity_search(query=query, k=4)

# Build context from retrieved chunks
context = "\n\n".join([doc.page_content for doc in res])

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't know based on the document."

Context:
{context}

Question: {question}

Answer:"""
)

# RAG chain: prompt → LLM
chain = prompt | models
response = chain.invoke({"context": context, "question": query})

print("\n--- Answer ---")
print(response.content)
