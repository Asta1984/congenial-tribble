from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import uuid
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Vector store and embeddings
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Text processing
import PyPDF2
import docx
from io import BytesIO
import re
from pathlib import Path

# LLM integration (using Ollama as free local option)
import httpx

# Document processing
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    embedding: Optional[np.ndarray] = None

class DocumentStore:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunks: List[DocumentChunk] = []
        self.index = None
        self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        
    def add_document(self, file_path: str, content: str, metadata: Dict[str, Any]):
        """Add a document to the store by chunking and embedding it"""
        chunks = self._chunk_text(content, metadata)
        
        for chunk in chunks:
            # Generate embedding
            embedding = self.embedding_model.encode(chunk.content)
            chunk.embedding = embedding
            self.chunks.append(chunk)
        
        # Rebuild FAISS index
        self._build_index()
        
    def _chunk_text(self, text: str, metadata: Dict[str, Any], chunk_size: int = 500, overlap: int = 50) -> List[DocumentChunk]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': len(chunks),
                'start_word': i,
                'end_word': min(i + chunk_size, len(words))
            })
            
            chunk = DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_id=str(uuid.uuid4())
            )
            chunks.append(chunk)
            
        return chunks
    
    def _build_index(self):
        """Build FAISS index from all chunks"""
        if not self.chunks:
            return
            
        embeddings = np.array([chunk.embedding for chunk in self.chunks])
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for similarity
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        self.index.add(embeddings)
    
    def search(self, query: str, k: int = 5) -> List[tuple]:
        """Search for relevant chunks"""
        if not self.index or not self.chunks:
            return []
            
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(distance)))
                
        return results
    
    def save_store(self, file_path: str):
        """Save document store to disk"""
        store_data = {
            'chunks': [
                {
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'chunk_id': chunk.chunk_id
                }
                for chunk in self.chunks
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(store_data, f, indent=2)
            
        # Save FAISS index
        if self.index:
            faiss.write_index(self.index, file_path.replace('.json', '.faiss'))
    
    def load_store(self, file_path: str):
        """Load document store from disk"""
        if not os.path.exists(file_path):
            return
            
        with open(file_path, 'r') as f:
            store_data = json.load(f)
            
        self.chunks = []
        for chunk_data in store_data['chunks']:
            chunk = DocumentChunk(
                content=chunk_data['content'],
                metadata=chunk_data['metadata'],
                chunk_id=chunk_data['chunk_id']
            )
            # Re-generate embeddings
            chunk.embedding = self.embedding_model.encode(chunk.content)
            self.chunks.append(chunk)
            
        # Load FAISS index
        faiss_path = file_path.replace('.json', '.faiss')
        if os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
        else:
            self._build_index()

class LLMClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize LLM client (default is Ollama)"""
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def generate_answer(self, query: str, context_chunks: List[DocumentChunk]) -> str:
        """Generate answer using retrieved context"""
        context = "\n\n".join([f"Source: {chunk.metadata.get('filename', 'Unknown')}\n{chunk.content}" 
                              for chunk in context_chunks])
        
        prompt = f"""Based on the following legal document excerpts, please provide a comprehensive answer to the user's question. 
Be precise and cite specific information from the provided sources.

Context:
{context}

Question: {query}

Answer:"""

        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": "tinyllama",  # You can change this to any model you have in Ollama
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "Unable to generate response")
            else:
                return f"Error generating response: {response.status_code}"
                
        except Exception as e:
            return f"Error connecting to LLM: {str(e)}"

class DocumentProcessor:
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file"""
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    @staticmethod
    def extract_text_from_docx(file_content: bytes) -> str:
        """Extract text from DOCX file"""
        doc = docx.Document(BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    @staticmethod
    def extract_text_from_txt(file_content: bytes) -> str:
        """Extract text from TXT file"""
        return file_content.decode('utf-8')

# Global instances
document_store = DocumentStore()
llm_client = LLMClient()
document_processor = DocumentProcessor()

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    max_results: int = 5

class Citation(BaseModel):
    text: str
    source: str
    chunk_id: str
    metadata: Dict[str, Any]
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    query: str
    timestamp: str

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int

# FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting RAG Backend...")
    
    # Load existing document store if available
    store_path = "document_store.json"
    if os.path.exists(store_path):
        document_store.load_store(store_path)
        print(f"Loaded {len(document_store.chunks)} chunks from existing store")
    
    # Create directories
    os.makedirs("uploads", exist_ok=True)
    
    yield
    
    # Shutdown
    print("Shutting down RAG Backend...")
    document_store.save_store("document_store.json")
    await llm_client.client.aclose()

app = FastAPI(
    title="Legal Document RAG Backend",
    description="A Retrieval-Augmented Generation backend for legal document queries",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a legal document"""
    try:
        # Read file content
        content = await file.read()
        
        # Extract text based on file type
        if file.filename.lower().endswith('.pdf'):
            text = document_processor.extract_text_from_pdf(content)
        elif file.filename.lower().endswith('.docx'):
            text = document_processor.extract_text_from_docx(content)
        elif file.filename.lower().endswith('.txt'):
            text = document_processor.extract_text_from_txt(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")
        
        # Save file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Add to document store
        metadata = {
            'filename': file.filename,
            'file_path': file_path,
            'upload_timestamp': datetime.now().isoformat(),
            'file_size': len(content)
        }
        
        chunks_before = len(document_store.chunks)
        document_store.add_document(file_path, text, metadata)
        chunks_created = len(document_store.chunks) - chunks_before
        
        return UploadResponse(
            message="Document uploaded and processed successfully",
            filename=file.filename,
            chunks_created=chunks_created
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the document store and generate an answer"""
    try:
        # Search for relevant chunks
        search_results = document_store.search(request.query, k=request.max_results)
        
        if not search_results:
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Extract chunks and create citations
        relevant_chunks = [chunk for chunk, score in search_results]
        citations = [
            Citation(
                text=chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,
                source=chunk.metadata.get('filename', 'Unknown'),
                chunk_id=chunk.chunk_id,
                metadata=chunk.metadata,
                relevance_score=score
            )
            for chunk, score in search_results
        ]
        
        # Generate answer using LLM
        answer = await llm_client.generate_answer(request.query, relevant_chunks)
        
        return QueryResponse(
            answer=answer,
            citations=citations,
            query=request.query,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    documents = defaultdict(lambda: {'chunks': 0, 'metadata': {}})
    
    for chunk in document_store.chunks:
        filename = chunk.metadata.get('filename', 'Unknown')
        documents[filename]['chunks'] += 1
        if not documents[filename]['metadata']:
            documents[filename]['metadata'] = {
                k: v for k, v in chunk.metadata.items() 
                if k not in ['chunk_index', 'start_word', 'end_word']
            }
    
    return {"documents": dict(documents)}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document and its chunks"""
    original_count = len(document_store.chunks)
    document_store.chunks = [
        chunk for chunk in document_store.chunks 
        if chunk.metadata.get('filename') != filename
    ]
    
    deleted_count = original_count - len(document_store.chunks)
    
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Rebuild index
    document_store._build_index()
    
    # Remove file
    file_path = f"uploads/{filename}"
    if os.path.exists(file_path):
        os.remove(file_path)
    
    return {"message": f"Deleted {deleted_count} chunks for document {filename}"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "documents_count": len(set(chunk.metadata.get('filename', 'Unknown') for chunk in document_store.chunks)),
        "chunks_count": len(document_store.chunks),
        "embedding_model": "all-MiniLM-L6-v2"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)