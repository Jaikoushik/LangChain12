"""
LangChain Document Loaders Reference Guide
This file provides quick code snippets for the most common document loaders in LangChain.
Make sure to install the required dependencies:
    pip install langchain-community beautifulsoup4 pypdf jq unstructured
"""

# -------------------------------------------------------------
# 1. TextLoader (Plain Text File)
# -------------------------------------------------------------
from langchain_community.document_loaders import TextLoader

def demo_text_loader():
    # Load a single text file
    loader = TextLoader("path/to/file.txt", encoding="utf-8")
    docs = loader.load()
    print(f"Loaded {len(docs)} text document(s)")
    return docs


# -------------------------------------------------------------
# 2. PyPDFLoader (PDF Documents)
# -------------------------------------------------------------
# Requires: pip install pypdf
from langchain_community.document_loaders import PyPDFLoader

def demo_pdf_loader():
    # PyPDFLoader splits the PDF into one document per page automatically
    loader = PyPDFLoader("path/to/document.pdf")
    pages = loader.load()
    print(f"Loaded {len(pages)} pages from PDF")
    # You can also load and split in one go
    # pages = loader.load_and_split()
    return pages


# -------------------------------------------------------------
# 3. CSVLoader (Comma-Separated Values)
# -------------------------------------------------------------
from langchain_community.document_loaders import CSVLoader

def demo_csv_loader():
    # CSVLoader loads each row in the CSV as a separate document
    loader = CSVLoader(
        file_path="path/to/data.csv",
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": ["Col1", "Col2", "Col3"]
        }
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} rows from CSV")
    return docs


# -------------------------------------------------------------
# 4. BSHtmlLoader (HTML Web Pages/Files)
# -------------------------------------------------------------
# Requires: pip install beautifulsoup4
from langchain_community.document_loaders import BSHtmlLoader

def demo_html_loader():
    # Uses BeautifulSoup4 to parse local HTML file and extract text and title metadata
    loader = BSHtmlLoader("path/to/page.html")
    docs = loader.load()
    print(f"Loaded HTML content")
    return docs


# -------------------------------------------------------------
# 5. JSONLoader (JSON Structured Data)
# -------------------------------------------------------------
# Requires: pip install jq
from langchain_community.document_loaders import JSONLoader

def demo_json_loader():
    # JSONLoader uses a jq schema to target specific fields/paths in the JSON structure
    # This example parses an array of messages and targets the 'content' field of each message
    loader = JSONLoader(
        file_path="path/to/data.json",
        jq_schema=".messages[].content",
        text_content=False
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} JSON items")
    return docs


# -------------------------------------------------------------
# 6. UnstructuredMarkdownLoader (Markdown Files)
# -------------------------------------------------------------
# Requires: pip install unstructured
from langchain_community.document_loaders import UnstructuredMarkdownLoader

def demo_markdown_loader():
    # UnstructuredMarkdownLoader parses headers, lists, and plain text from Markdown files
    loader = UnstructuredMarkdownLoader("path/to/README.md")
    docs = loader.load()
    print(f"Loaded markdown document")
    return docs


# -------------------------------------------------------------
# 7. WebBaseLoader (Fetch/Scrape from Live Web URL)
# -------------------------------------------------------------
# Requires: pip install beautifulsoup4
from langchain_community.document_loaders import WebBaseLoader

def demo_web_base_loader():
    # Scrapes HTML content directly from a live URL
    loader = WebBaseLoader("https://www.example.com")
    docs = loader.load()
    print(f"Scraped content from web page")
    return docs


# -------------------------------------------------------------
# 8. DirectoryLoader (Load all files from a directory)
# -------------------------------------------------------------
# Requires: pip install unstructured
from langchain_community.document_loaders import DirectoryLoader

def demo_directory_loader():
    # Load all files matching a glob pattern, specifying which loader class to use
    loader = DirectoryLoader(
        "path/to/dir",
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from directory")
    return docs