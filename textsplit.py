"""
LangChain Text Splitters Reference Guide
This file provides quick code snippets for the most common text splitters in LangChain.
These are used to chunk large documents into smaller pieces before indexing them in a vector database.

Make sure to install the required dependencies:
    pip install langchain-text-splitters tiktoken
"""

# -------------------------------------------------------------
# 1. RecursiveCharacterTextSplitter (Recommended Default)
# -------------------------------------------------------------
# Splits text by a list of characters (default: ["\n\n", "\n", " ", ""]) recursively.
# It tries to keep paragraphs, sentences, and words together as much as possible.
from langchain_text_splitters import RecursiveCharacterTextSplitter

def demo_recursive_character_splitter():
    text = (
        "LangChain is a framework for developing applications powered by language models. "
        "It enables applications that are context-aware and reason based on the provided context. "
        "To build these, we often need to chunk large pieces of text into smaller, manageable parts. "
        "This is where text splitters come in handy. They help maintain semantic meaning by keeping "
        "related sentences and paragraphs together."
    )
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,         # Maximum size of each chunk (in characters/tokens depending on length_function)
        chunk_overlap=20,       # Number of characters to overlap between adjacent chunks
        length_function=len,    # Function to calculate chunk size (default: len)
        is_separator_regex=False
    )
    
    # split_text returns raw strings
    chunks = splitter.split_text(text)
    print(f"--- RecursiveCharacterTextSplitter ---")
    print(f"Created {len(chunks)} chunks:")
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx+1}: {repr(chunk)}")
        
    # Alternatively, create Document objects (with metadata)
    # docs = splitter.create_documents([text], metadatas=[{"source": "demo"}])
    return chunks


# -------------------------------------------------------------
# 2. CharacterTextSplitter (Split by a specific separator)
# -------------------------------------------------------------
# Splits text on a specific character or sequence (default: "\n\n").
# Note: It will only split if a chunk exceeds chunk_size AND the separator is found.
from langchain_text_splitters import CharacterTextSplitter

def demo_character_splitter():
    text = "Paragraph 1 line A\nParagraph 1 line B\n\nParagraph 2 line A\nParagraph 2 line B"
    
    splitter = CharacterTextSplitter(
        separator="\n\n",       # The specific character to split on
        chunk_size=30,          # Target chunk size
        chunk_overlap=10,       # Overlap size
        length_function=len
    )
    
    chunks = splitter.split_text(text)
    print(f"\n--- CharacterTextSplitter ---")
    print(f"Created {len(chunks)} chunks:")
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx+1}: {repr(chunk)}")
    return chunks


# -------------------------------------------------------------
# 3. TokenTextSplitter (Split by token count)
# -------------------------------------------------------------
# Splits text based on the number of tokens (words/sub-words as seen by LLMs).
# Useful for ensuring chunks do not exceed LLM context window token limits.
# Requires: pip install tiktoken
from langchain_text_splitters import TokenTextSplitter

def demo_token_splitter():
    text = (
        "Tokenization is the process of converting text into tokens. "
        "LLMs have a maximum limit of tokens they can process in a single request."
    )
    
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base", # Encoding used by gpt-4 / gpt-3.5
        chunk_size=10,               # Number of tokens per chunk
        chunk_overlap=2              # Overlap of tokens between chunks
    )
    
    chunks = splitter.split_text(text)
    print(f"\n--- TokenTextSplitter ---")
    print(f"Created {len(chunks)} chunks:")
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx+1}: {repr(chunk)}")
    return chunks


# -------------------------------------------------------------
# 4. MarkdownHeaderTextSplitter (Split markdown by headers)
# -------------------------------------------------------------
# Splits markdown content by headers (e.g., #, ##, ###) and moves the header
# titles into the chunk's metadata. This preserves the context/section of each chunk.
from langchain_text_splitters import MarkdownHeaderTextSplitter

def demo_markdown_header_splitter():
    markdown_document = (
        "# LangChain Guide\n\n"
        "This is the introduction.\n\n"
        "## Document Loaders\n\n"
        "Loaders help read files.\n\n"
        "### TextLoader\n\n"
        "Loads plain text files.\n\n"
        "## Text Splitters\n\n"
        "Splitters chunk text."
    )
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True # Removes the headers from the chunk text since they are in metadata
    )
    
    docs = splitter.split_text(markdown_document)
    print(f"\n--- MarkdownHeaderTextSplitter ---")
    print(f"Created {len(docs)} document chunks:")
    for idx, doc in enumerate(docs):
        print(f"Chunk {idx+1} Metadata: {doc.metadata}")
        print(f"Chunk {idx+1} Content: {repr(doc.page_content)}")
    return docs


# -------------------------------------------------------------
# 5. RecursiveJsonSplitter (Split JSON data structure)
# -------------------------------------------------------------
# Splits nested JSON data structures into smaller JSON objects or text chunks while
# keeping nested keys/path structure intact.
from langchain_text_splitters import RecursiveJsonSplitter

def demo_json_splitter():
    json_data = {
        "company": "OpenAI",
        "models": {
            "gpt-4": {"tokens": 128000, "type": "chat"},
            "gpt-3.5-turbo": {"tokens": 16385, "type": "chat"}
        },
        "pricing": {
            "gpt-4-input": 0.03,
            "gpt-4-output": 0.06
        }
    }
    
    splitter = RecursiveJsonSplitter(max_chunk_size=100)
    
    # Split the JSON object into smaller JSON objects
    json_chunks = splitter.split_json(json_data)
    
    # Or split JSON into langchain Document objects
    docs = splitter.create_documents(texts=[json_data])
    
    print(f"\n--- RecursiveJsonSplitter ---")
    print(f"Created {len(json_chunks)} JSON chunks:")
    for idx, chunk in enumerate(json_chunks):
        print(f"Chunk {idx+1}: {chunk}")
    return json_chunks


if __name__ == "__main__":
    demo_recursive_character_splitter()
    demo_character_splitter()
    demo_token_splitter()
    demo_markdown_header_splitter()
    demo_json_splitter()
