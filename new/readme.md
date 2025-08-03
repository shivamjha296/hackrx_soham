# LLM-Powered Intelligent Query–Retrieval System 🤖📄

This project is an intelligent query-retrieval system designed to process large documents and answer natural language questions with high accuracy. It leverages a **Retrieval-Augmented Generation (RAG)** architecture to provide contextually-aware and explainable answers, making it ideal for domains like insurance, legal, HR, and compliance.

-----

## ✨ Features

- **Multi-Format Document Processing:** Handles PDF documents fetched directly from a URL.
- **Semantic Search:** Uses advanced `BAAI/bge-base-en-v1.5` embeddings and a FAISS vector index for fast and accurate clause retrieval.
- **LLM-Powered Decision Making:** Integrates with the **Mistral** LLM to generate precise answers based on retrieved document context.
- **Explainable AI:** Answers are grounded in the provided text, ensuring traceability and minimizing hallucinations.
- **Structured Output:** Delivers responses in a clean, predictable JSON format.
- **Secure API:** The endpoint is protected via Bearer Token authentication.

-----

## 🛠️ Tech Stack

- **Backend Framework:** FastAPI
- **Large Language Model (LLM):** Mistral (`mistral-large-latest`)
- **Embedding Model:** BAAI/bge-base-en-v1.5
- **Vector Search:** FAISS (via `sentence-transformers`)
- **Document Parsing:** PyPDF

-----

## ⚙️ Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Prerequisites

- Python 3.9+
- A Mistral API Key

### 2. Installation

First, clone the repository or download the project files. Then, navigate to the project directory and set up the environment.

```bash
# Navigate into the project folder
cd /path/to/project

# Create and activate a virtual environment
python -m venv venv
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```

### 3. Configuration

The application requires a Mistral API key to function.

- Rename the `env.example` file to `.env`.
- Open the newly created `.env` file and insert your Mistral API key:
  ```
  MISTRAL_API_KEY="your_secret_mistral_api_key_here"
  ```

-----

## ▶️ Running the Application

Once the setup is complete, you can start the API server using Uvicorn.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will now be live and accessible at `http://localhost:8000`.

-----

## 🧪 API Usage

You can interact with the API by sending a `POST` request to the `/hackrx/run` endpoint.

- **Endpoint:** `POST /hackrx/run`
- **Authentication:** `Authorization: Bearer 02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0`

### Sample cURL Request (macOS/Linux)

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
-H "Content-Type: application/json" \
-H "Accept: application/json" \
-H "Authorization: Bearer 02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0" \
-d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for cataract surgery?"
    ]
}'
```

### Sample PowerShell Request (Windows)

```powershell
Invoke-WebRequest -Uri "http://localhost:8000/hackrx/run" `
    -Method POST `
    -Headers @{
        "Content-Type"  = "application/json";
        "Accept"        = "application/json";
        "Authorization" = "Bearer 02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0"
    } `
    -Body '{
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for cataract surgery?"
        ]
    }'
```

### Sample JSON Response

A successful request will return a `200 OK` status with a JSON body containing the answers.

```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
    "The policy has a specific waiting period of two (2) years for cataract surgery."
  ]
}
```