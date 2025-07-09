import os

# Define folder and file structure
structure = {
    "rag_backend": {
        "app": {
            "api": {
                "routes_upload.py": "",
                "routes_query.py": "",
                "routes_documents.py": "",
                "routes_health.py": "",
            },
            "core": {
                "document_processor.py": "",
                "document_store.py": "",
                "llm_client.py": "",
            },
            "models": {
                "base.py": "",
                "query.py": "",
            },
            "main.py": "# Entry point for FastAPI app\n\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n# Include routers here\n"
        },
        "uploads": {},  # folder for storing uploaded files
        "config.py": "# Configuration settings",
        "requirements.txt": """fastapi
uvicorn
httpx
pydantic
sentence-transformers
faiss-cpu
PyPDF2
python-docx
"""
    }
}


def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)


if __name__ == "__main__":
    create_structure(".", structure)
    print("✅ RAG backend project structure created.")
