# Dynamic Chunking RAG System

A production-ready Retrieval-Augmented Generation (RAG) system with intelligent document-aware chunking, FAISS indexing, and Mistral-7B generation. Features structure-aware parsing that preserves document hierarchy and semantic boundaries.

## Overview

This notebook implements:
- **Intelligent Document Chunking**: Structure-aware parsing that respects headings, paragraphs, and lists
- **Hierarchical Section Tracking**: Maintains document structure throughout chunking process
- **FAISS Vector Search**: GPU-accelerated similarity search with IVF indexing
- **Sentence-Level Splitting**: Prevents mid-sentence breaks in chunks
- **Mistral-7B Generation**: High-quality answer generation with retrieved context
- **Stress Testing Framework**: Benchmarks performance on documents up to 10M characters

## Key Features

### Dynamic Chunking System
- **Structure Detection**: Automatically identifies headings (markdown #, numbered, ALL CAPS)
- **Semantic Boundaries**: Respects paragraphs, lists, and section boundaries
- **Configurable Limits**: Max/min chunk sizes with intelligent overflow handling
- **Sentence Preservation**: Splits long paragraphs at sentence boundaries
- **Section Context**: Tracks hierarchical path (e.g., "1.2.3 Introduction > Methods")

### Vector Search
- **FAISS IVF Index**: Efficient approximate nearest neighbor search
- **GPU Acceleration**: Automatic GPU utilization when available
- **Normalized Embeddings**: L2-normalized for cosine similarity via inner product
- **Adaptive Parameters**: Dynamic nlist adjustment based on corpus size
- **Fallback Strategy**: Uses flat index for small datasets (<64 chunks)

### RAG Pipeline
- **Batch Embedding**: Efficient encoding with sentence transformers
- **Top-K Retrieval**: Configurable number of relevant chunks
- **Context Assembly**: Smart truncation to fit model context window
- **Mistral-7B Instruct**: State-of-the-art generation with proper prompting
- **Source Attribution**: Includes document ID, chunk ID, and section path

## Installation

All dependencies are installed automatically in the notebook:

```bash
pip install sentence-transformers faiss-gpu transformers torch accelerate safetensors psutil
```

**Note**: Use `faiss-cpu` if GPU is not available.

## Quick Start

### 1. Basic RAG Setup

```python
from embedder import Embedder
from rag_engine import RAGEngine

embedder = Embedder()
rag = RAGEngine(embedder)

docs = {
    "doc_1": """
    1. Introduction
    This document describes our system architecture.
    
    1.1 Background
    We use microservices with event-driven communication.
    
    2. Implementation
    The system is built on Kubernetes with Istio service mesh.
    """,
    "doc_2": """
    # API Documentation
    
    ## Authentication
    All endpoints require Bearer token authentication.
    
    ## Rate Limiting
    Maximum 100 requests per minute per API key.
    """
}

rag.build_from_docs(docs)
```

### 2. Query the System

```python
question = "What authentication method is required?"
answer = rag.answer(question, top_k=6)
print(answer)
```

### 3. Run Stress Tests

```python
stress_test(1_000_000)  # Test with 1M character document
stress_test(10_000_000)  # Test with 10M character document
```

## System Architecture

```
Document Text
    ↓
[Structure Detection] → Headings, Paragraphs, Lists
    ↓
[Sentence Splitting] → Break long paragraphs at sentence boundaries
    ↓
[Dynamic Chunking] → Respect structure + size constraints
    ↓
[Embedding] → Sentence Transformers (384-dim)
    ↓
[FAISS Indexing] → IVF with GPU acceleration
    ↓
[Query] → Semantic search + Top-K retrieval
    ↓
[Generation] → Mistral-7B with context
    ↓
Answer
```

## Configuration Parameters

### Chunking
```python
MAX_CHARS_PER_CHUNK = 1500      # Maximum characters per chunk
MIN_CHARS_PER_CHUNK = 400       # Minimum characters per chunk
MAX_CHARS_PER_SUBUNIT = 700     # Max chars before sentence split
```

### Embedding
```python
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
DEVICE = "cuda"  # or "cpu"
```

### FAISS Index
```python
FAISS_TARGET_NLIST = 256        # IVF clusters for search
FAISS_USE_GPU_IF_AVAILABLE = True
```

### Generation
```python
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
RAG_MAX_CONTEXT_CHARS = 8000    # Max context for LLM
```

## Core Components

### 1. Structure Detection
```python
def detect_heading_level(line: str) -> Optional[int]:
    """Detect heading level from:
    - Markdown: # (level 1), ## (level 2), etc.
    - Numbered: 1.2.3 (level 3)
    - ALL CAPS: SECTION TITLE (level 2)
    """
```

### 2. Document Segmentation
```python
def segment_into_units(text: str) -> List[Unit]:
    """Parse document into structured units:
    - Headings with hierarchy level
    - Paragraphs with complete text
    - List items (bullet or numbered)
    """
```

### 3. Intelligent Chunking
```python
def chunk_document(units: List[Unit], doc_id: str) -> List[Chunk]:
    """Create chunks that:
    - Respect section boundaries (new heading = new chunk)
    - Stay within size limits (400-1500 chars)
    - Preserve hierarchical context
    - Never split mid-sentence
    """
```

### 4. RAG Engine
```python
class RAGEngine:
    def build_from_docs(self, docs: Dict[str, str]):
        """Index documents with:
        1. Structure-aware chunking
        2. Batch embedding
        3. FAISS index construction
        """
    
    def answer(self, question: str, top_k: int = 6) -> str:
        """Generate answer from:
        1. Semantic search (top-k chunks)
        2. Context assembly with metadata
        3. Mistral-7B generation
        """
```

## Data Structures

### Unit
```python
@dataclass
class Unit:
    type: str                    # "heading", "paragraph", "list"
    text: str                    # Content
    level: Optional[int] = None  # Heading level (1-6)
```

### Chunk
```python
@dataclass
class Chunk:
    doc_id: str                  # Source document
    chunk_id: str                # Unique identifier
    section_path: List[str]      # ["1. Intro", "1.2 Methods"]
    units: List[Unit]            # Structured content
    char_len: int                # Character count
    position: int                # Chunk index in document
    embedding: np.ndarray        # 384-dim vector
```

## Stress Testing

The notebook includes comprehensive stress tests:

```python
sizes = [
    100_000,    # 100K chars (~20 pages)
    1_000_000,  # 1M chars (~200 pages)
    5_000_000,  # 5M chars (~1000 pages)
    10_000_000  # 10M chars (~2000 pages)
]
```

### Metrics Tracked
- **Segmentation Time**: Parse into units
- **Chunking Time**: Create chunks with boundaries
- **Embedding Time**: Encode all chunks
- **FAISS Build Time**: Index construction
- **Retrieval Time**: Search latency
- **Memory Usage**: Peak RAM consumption

### Actual Performance (T4 GPU - Stress Test Results)

| Size | Chunks | Embed Time | Index Time | Search Time | Memory | Throughput |
|------|--------|------------|------------|-------------|--------|------------|
| 100K | 76 | 0.17s | 0.03s | 0.01s | 18 MB | 476K chars/s |
| 1M | 760 | 1.70s | 0.01s | 0.01s | 19 MB | 437K chars/s |
| 5M | 3,800 | 8.46s | 0.09s | 0.01s | 48 MB | 218K chars/s |
| 10M | 7,599 | 16.97s | 0.19s | 0.01s | 70 MB | 135K chars/s |

**Key Observations**:
- Consistent embedding throughput: ~450 chunks/second
- Sub-second indexing even at 10M chars
- Instant retrieval: 0.01s regardless of corpus size
- Excellent memory efficiency: 70 MB for 10M characters

## Advanced Features

### Custom Heading Detection
```python
HEADING_RE_NUM = re.compile(r"^\s*\d+(\.\d+)*\s+")      # "1.2.3 Title"
HEADING_RE_HASH = re.compile(r"^\s*#{1,6}\s+")          # "## Title"
HEADING_RE_ALLCAPS = re.compile(r"^[A-Z0-9 ,;:\-]{8,}$") # "SECTION TITLE"
```

### Sentence Splitting
```python
def split_into_sentences(text: str) -> List[str]:
    """Split at sentence boundaries (.!?) while preserving:
    - Dr., Mr., Ms. (common abbreviations)
    - Decimal numbers (3.14)
    - Multiple punctuation (!!!, ...)
    """
```

### Context Assembly
```python
def generate_answer_rag(question: str, retrieved: List[Tuple[Chunk, float]]):
    """Assemble context with:
    - Document ID and chunk ID
    - Section hierarchy path
    - Similarity score
    - Truncation at RAG_MAX_CONTEXT_CHARS
    """
```

## Optimization Tips

### Improving Quality
1. **Tune Chunk Size**: Larger chunks (1500-2000) for broader context
2. **Increase Top-K**: Retrieve more chunks (8-10) for complex questions
3. **Better Embeddings**: Use "all-mpnet-base-v2" (768-dim) instead of MiniLM
4. **Adjust nprobe**: Higher nprobe (√nlist) for better recall

### Reducing Latency
1. **Smaller Embeddings**: Stick with MiniLM-L6-v2 (384-dim)
2. **Lower Top-K**: Retrieve fewer chunks (3-5)
3. **Reduce nlist**: Fewer IVF clusters (64-128)
4. **CPU Fallback**: Use faiss-cpu for small datasets

### Memory Optimization
1. **Batch Size**: Reduce embedding batch_size to 32
2. **Lazy Loading**: Don't load Mistral until needed
3. **Clear Cache**: Delete embedder after indexing
4. **Float16**: Use half precision for GPU


## Troubleshooting

**Issue**: FAISS build fails
**Solution**: Check if GPU has enough memory; fallback to CPU or reduce batch size

**Issue**: Poor retrieval quality
**Solution**: Increase chunk_size or adjust MIN_CHARS_PER_CHUNK

**Issue**: Mistral OOM error
**Solution**: Reduce RAG_MAX_CONTEXT_CHARS or use int8 quantization

**Issue**: Slow embedding
**Solution**: Increase batch_size or use GPU

## System Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (T4, V100, A100)
- **RAM**: 16GB+ system memory
- **Python**: 3.8+
- **CUDA**: 11.8+ for GPU acceleration

## File

- `dynamic-chunk.ipynb`: Complete implementation with all components and stress tests


## License

This project is provided for research and educational purposes.

## Acknowledgments

- Sentence Transformers for efficient embeddings
- FAISS for high-performance vector search
- Mistral AI for instruction-tuned LLM
- PyTorch and Transformers ecosystem
