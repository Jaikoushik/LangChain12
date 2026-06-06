from langchain_text_splitters import RecursiveCharacterTextSplitter

def demo_recursive_character_splitter():
    text = (
        "LangChain is a framework for developing applications powered by language models. "
        "It enables applications that are context-aware and reason based on the provided context. "
        " To build these, we often need to chunk large pieces of text into smaller, manageable parts. "
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

if __name__ == "__main__":
    demo_recursive_character_splitter()