# Legal Document RAG System

A Retrieval-Augmented Generation (RAG) system for legal documents using FastAPI, ChromaDB, and Ollama for local LLM inference.

## Features

- Document upload and processing (PDF, DOCX, TXT)
- Semantic search using ChromaDB vector database
- Local LLM inference using Ollama
- RESTful API with automatic documentation
- Citation tracking and relevance scoring

## Prerequisites

- Python 3.8+
- Git
- At least 8GB RAM (16GB recommended for larger models)
- 10GB+ free disk space for models

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Asta1984/lexi.sg-rag-backend-test.git
cd legal-document-rag
```

### 2. Install Python Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install and Setup Ollama

#### Option A: Using Ollama Installer (Recommended)

1. **Download Ollama:**
   - Visit [ollama.ai](https://ollama.ai)
   - Download the installer for your OS
   - Run the installer

2. **Verify Installation:**
   ```bash
   ollama --version
   ```

#### Option B: Using Package Managers

**macOS (Homebrew):**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai](https://ollama.ai) and run the installer.

### 4. Install Local LLM Models

#### Start Ollama Service

```bash
# Start Ollama server (if not auto-started)
ollama serve
```

#### Install Recommended Models

**Option 1: TinyLlama (Fastest, ~637MB)**
```bash
ollama pull tinyllama
```

**Option 2: Llama 3.2 1B (Better quality, ~1.3GB)**
```bash
ollama pull llama3.2:1b
```

**Option 3: Gemma 2B (Good balance, ~1.6GB)**
```bash
ollama pull gemma:2b
```

**Option 4: Llama 3.2 3B (Best quality, ~2GB)**
```bash
ollama pull llama3.2:3b
```

#### Verify Model Installation

```bash
# List installed models
ollama list

# Test model
ollama run tinyllama 
```

### 5. Configure the Application

1. **Update Model Configuration:**
   Edit your `main.py` or configuration file to use your preferred model:
   ```python
   # In LLMClient class
   "model": "tinyllama",  # or "llama3.2:1b", "gemma:2b", etc.
   ```

2. **Create Required Directories:**
   ```bash
   mkdir -p uploads
   mkdir -p chroma_db
   ```

### 6. Run the Application

```bash
# Start the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- Main API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## API Testing

### 1. Test Ollama Connection

First, verify Ollama is running:

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags



### 2. Test API Endpoints

#### Upload a Document

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/your/document.pdf"
```

#### List Uploaded Documents

```bash
curl -X GET "http://localhost:8000/documents"
```

#### Query Documents

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this case about?",
    "max_results": 5
  }'
```

#### Search Documents (Without LLM)

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "contract terms",
    "max_results": 10
  }'
```

#### Delete a Document

```bash
curl -X DELETE "http://localhost:8000/documents/your-document-name.pdf"
```

### 3. Using the Interactive API Documentation

1. Navigate to http://localhost:8000/docs
2. Click "Try it out" on any endpoint
3. Fill in the parameters
4. Click "Execute"

### 4. Sample Test Workflow

```bash
# 1. Upload a legal document
curl -X POST "http://localhost:8000/upload" \
  -F "file=@sample_legal_document.pdf"

# 2. Verify upload
curl -X GET "http://localhost:8000/documents"

# 3. Query the document
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main legal issues discussed?",
    "max_results": 3
  }'
```

## Troubleshooting

### Common Issues

#### 1. "Error connecting to LLM: ReadTimeout"

**Cause:** Model is taking too long to respond.

**Solutions:**
- Use a smaller model (tinyllama)
- Increase timeout in code
- Reduce context length
- Check system resources

#### 2. "Cannot connect to Ollama server"

**Cause:** Ollama service not running.

**Solutions:**
```bash
# Start Ollama
ollama serve

# Or restart the service
# On macOS/Linux:
sudo systemctl restart ollama

# On Windows:
# Restart Ollama from the system tray
```

#### 3. "Model not found"

**Cause:** Model not installed.

**Solutions:**
```bash
# Check installed models
ollama list

# Install missing model
ollama pull tinyllama
```

#### 4. Memory Issues

**Cause:** Not enough RAM for the model.

**Solutions:**
- Use smaller models (tinyllama, llama3.2:1b)
- Close other applications
- Increase system swap space

### Performance Tips

1. **Model Selection:**
   - Development: Use `tinyllama` for fastest responses
   - Production: Use `llama3.2:3b` for better quality
   
2. **Hardware Optimization:**
   - Use SSD storage for faster model loading
   - Ensure adequate RAM (8GB+ recommended)
   - Close unnecessary applications

3. **Query Optimization:**
   - Keep queries specific and focused
   - Use smaller `max_results` values for faster responses
   - Consider caching frequent queries

## Configuration Options

### Environment Variables

Create a `.env` file:

```env
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=tinyllama
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
VECTOR_DB_PATH=./chroma_db
UPLOAD_DIR=./uploads
```

### Model Configuration

Edit the model settings in your code:

```python
# For faster responses
"options": {
    "temperature": 0.1,
    "num_predict": 200,    # Limit response length
    "top_k": 10,
    "top_p": 0.9
}

# For better quality
"options": {
    "temperature": 0.3,
    "num_predict": 500,
    "top_k": 40,
    "top_p": 0.95
}
```

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/upload` | Upload a document |
| GET | `/documents` | List uploaded documents |
| POST | `/query` | Query documents with LLM |
| POST | `/search` | Search documents (vector search only) |
| DELETE | `/documents/{filename}` | Delete a document |
| GET | `/health` | Health check |

### Response Format

```json
{
  "answer": "Generated answer from LLM",
  "citations": [
    {
      "text": "Relevant excerpt from document",
      "source": "document_name.pdf",
      "chunk_id": "unique-chunk-id",
      "metadata": {
        "filename": "document_name.pdf",
        "file_path": "uploads/document_name.pdf",
        "chunk_index": 1
      },
      "relevance_score": 0.85
    }
  ],
  "query": "Original query",
  "timestamp": "2025-07-10T15:30:00.123456"
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues on GitHub
3. Create a new issue with detailed information

---

**Note:** This system is designed for legal document analysis and should not be used as a substitute for professional legal advice.