# SmartSim Backend

The **FastAPI** backend for SmartSim: an image similarity search and clustering API powered by **OpenAI CLIP** embeddings and **SQLite + sqlite-vec** vector search.

Upload a folder of images, and the API embeds each one with CLIP and stores the vectors in SQLite. You can then search the collection by **text** ("a dog on the beach") or by **image**, or group it into **clusters** of visually similar pictures.

> Frontend: [smartsim-frontend](https://github.com/SifeYassine/smartsim-frontend) · Original desktop version: [smartsim](https://github.com/SifeYassine/smartsim)

## Features

- **Bulk folder upload** with validation, de-duplication (already-indexed files are skipped), and automatic embedding
- **Single image upload** for use as a search query
- **Text → image search** using CLIP text embeddings
- **Image → image search** using CLIP image embeddings
- **Native vector search** in SQLite with the `sqlite-vec` extension (`vec_distance_cosine`)
- **Hierarchical clustering** (agglomerative, average linkage, cosine distance) by number of clusters or similarity threshold, with min/max cluster-size filters; results are saved back to the database
- **Static file serving** of uploaded images at `/static`
- Fully **async** stack (SQLAlchemy async + aiosqlite + aiofiles)
- Interactive API docs via Swagger UI

## Tech Stack

| Concern       | Technology |
| ------------- | ---------- |
| API           | FastAPI, Uvicorn, Pydantic v2 / pydantic-settings |
| ML            | PyTorch, Hugging Face Transformers (`openai/clip-vit-base-patch16`) |
| Database      | SQLite, SQLAlchemy (async), aiosqlite, sqlite-vec |
| Clustering    | scikit-learn |
| Files         | aiofiles, python-multipart |

## Project Structure

```
smartsim-backend/
├── app/
│   ├── main.py                    # App factory, CORS, static files, routers
│   ├── api/
│   │   ├── images.py              # /images endpoints
│   │   └── search.py              # /search endpoints
│   ├── core/
│   │   ├── config.py              # Settings (env-overridable)
│   │   └── deps.py                # Dependency injection
│   ├── crud/image.py              # DB queries incl. vector search
│   ├── db/                        # Engine, session, sqlite-vec loading, Vector type
│   ├── models/image.py            # Image table
│   ├── schemas/                   # Request/response models + validation
│   └── services/
│       ├── embedding_service.py   # CLIP image/text embeddings
│       ├── similarity_service.py  # Text & image search
│       ├── clustering_service.py  # Agglomerative clustering
│       ├── file_storage_service.py
│       └── image_upload_service.py
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.10+
- A Python build whose `sqlite3` module supports loading extensions (required by `sqlite-vec`)

### Installation

```bash
git clone https://github.com/SifeYassine/smartsim-backend.git
cd smartsim-backend

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration (optional)

All settings have defaults and can be overridden in a `.env` file:

| Variable            | Default                                   | Description |
| ------------------- | ----------------------------------------- | ----------- |
| `DATABASE_URL`      | `sqlite+aiosqlite:///db/smartsim.db`      | Database connection |
| `DB_FILE_PATH`      | `./db`                                    | Database directory |
| `STATIC_FILES_PATH` | `./static/images`                         | Where uploaded images are stored |
| `MODEL_NAME`        | `openai/clip-vit-base-patch16`            | CLIP model to use |
| `MODEL_CACHE_DIR`   | `./cache`                                 | Model download cache |
| `CORS_ORIGINS`      | `["http://localhost:3000","http://localhost:3001"]` | Allowed frontend origins |

> The [frontend](https://github.com/SifeYassine/smartsim-frontend) dev server runs on port **3000**, which is allowed by default.

### Run

```bash
uvicorn app.main:app --reload
```

The API runs at <http://127.0.0.1:8000>. Interactive docs are at **`/docs`**.
On first start, the CLIP model is downloaded to `./cache`.

## API Reference

### Images

| Method | Endpoint                | Description |
| ------ | ----------------------- | ----------- |
| POST   | `/images/upload-folder` | Upload multiple images (`files`, multipart) and index them |
| POST   | `/images/upload-image`  | Upload a single image (`file`, multipart); returns its `file_path` |
| DELETE | `/images/clear`         | Delete all images and their embeddings |

Accepted formats: `png`, `jpg`, `jpeg`, `jfif`, `webp`.

### Search

| Method | Endpoint          | Body |
| ------ | ----------------- | ---- |
| POST   | `/search/text`    | `{ "query": "a red car", "top_k": 5, "similarity_threshold": 0.12 }` |
| POST   | `/search/image`   | `{ "image_path": "<file_path from upload-image>", "top_k": 5, "similarity_threshold": 0.12 }` |
| POST   | `/search/cluster` | `{ "max_num_clusters": 5 }` **or** `{ "similarity_threshold": 0.8 }`, optionally with `min_images_per_cluster` / `max_images_per_cluster` |

`top_k` must be 1–100 and `similarity_threshold` 0.0–1.0.

### Health

| Method | Endpoint  | Description |
| ------ | --------- | ----------- |
| GET    | `/`       | API status message |
| GET    | `/health` | Health check |

## Example

```bash
# Index a few images
curl -X POST http://127.0.0.1:8000/images/upload-folder \
  -F "files=@cat1.jpg" -F "files=@cat2.jpg" -F "files=@dog.jpg"

# Search by text
curl -X POST http://127.0.0.1:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "a sleeping cat", "top_k": 3}'
```

## Author

**Sifeddine Yassine** · [@SifeYassine](https://github.com/SifeYassine)
