
## Linux documentation RAG agent (`linux-doc-rag.py`)

Ask natural-language questions about Linux tools and get accurate answers
sourced directly from locally installed man pages.  The agent uses
**Retrieval-Augmented Generation (RAG)**: man pages are split into chunks,
embedded into a local vector database (ChromaDB), and at query time the most
relevant chunks are retrieved and passed as context to a local LLM via
[Ollama](https://ollama.com/).  No internet connection or API keys are needed.

### Prerequisites

```bash
pip install chromadb          # vector database
ollama serve                  # start the Ollama daemon
ollama pull nomic-embed-text  # embedding model (used during ingest & query)
ollama pull llama3.2          # (or any other chat model you prefer)
```

### Quick start

```bash
# 1. Index all installed man pages (runs once; may take several minutes for
#    large installations – progress is printed every 200 pages).
./linux-doc-rag.py ingest

# 2. Ask a question:
./linux-doc-rag.py query "How do I find all files modified in the last 24 hours?"

# 3. Interactive mode:
./linux-doc-rag.py query --interactive

# 4. Show which man-page sections were used to build the answer:
./linux-doc-rag.py query --show-sources "How do I change file ownership recursively?"
```

### Ingest options

| Flag | Default | Description |
|---|---|---|
| `--man-dirs DIR …` | `/usr/share/man` | Man page directories to scan |
| `--doc-dirs DIR …` | *(none)* | Additional doc dirs, e.g. `/usr/share/doc` (README, CHANGELOG, *.md) |
| `--sections N …` | all | Only index specific sections (e.g. `--sections 1 8`) |
| `--force` | off | Re-embed already-indexed entries |
| `--embed-model MODEL` | `nomic-embed-text` | Ollama embedding model |
| `--ollama-url URL` | `http://localhost:11434` | Ollama base URL |
| `--db-path DIR` | `~/.local/share/linux-doc-rag/chroma` | ChromaDB storage path |
| `--batch-size N` | `50` | Write batch size |

### Query options

| Flag | Default | Description |
|---|---|---|
| `--interactive` / `-i` | off | Enter interactive Q&A loop |
| `--chat-model MODEL` | `llama3.2` | Ollama chat/completion model |
| `--top-k N` | `6` | Number of context chunks to retrieve |
| `--show-sources` | off | Print the retrieved source chunks |
| `--embed-model MODEL` | `nomic-embed-text` | Ollama embedding model |
| `--ollama-url URL` | `http://localhost:11434` | Ollama base URL |
| `--db-path DIR` | `~/.local/share/linux-doc-rag/chroma` | ChromaDB storage path |

### Tips

* **Section filtering** dramatically reduces ingest time.  Sections 1 (user
  commands) and 8 (admin commands) cover the most useful material:
  ```bash
  ./linux-doc-rag.py ingest --sections 1 8
  ```
* **Resume support**: re-running `ingest` skips already-indexed chunks, so
  you can safely interrupt and restart.
* **Multiple man trees**: pass several `--man-dirs` to combine the system
  tree with locale-specific or third-party trees.
* **/usr/share/doc**: add `--doc-dirs /usr/share/doc` to also index README
  and CHANGELOG files for installed packages (secondary to man pages).

---

## Syncing Ollama models with Llama.cpp and llama-swap

Let's say you want to run either Ollama or llama.cpp (via llama-swap),
but you don't want two different copies of the models. Here's an easy
way to reuse the models across tools:

1. Pull your models with `ollama pull <model url>`

2. Update the llama.cpp cache with symlinks from ollama's model cache.

   On a Linux host, ollama's models get downloaded to an Ollama user's
   home dir, which is /usr/shre/ollama/.
   
   ```bash
   $ ./ollama-to-llamacpp-modelcache.sh -o /usr/share/ollama/.ollama/models
   ```

3. Update llama-swap config file based on llama.cpp cache.

   ```bash
   $ ./llama-swap-config-gen.py --verbose --prune-missing
   ```

4. If it's not already running, run `llama-swap`:

   ```bash
   $ ./llama-swap.sh
   ```

