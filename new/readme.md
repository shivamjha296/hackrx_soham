# LLM-Powered Intelligent Query–Retrieval System 🤖📄

This project is an intelligent query-retrieval system designed to process large documents and answer natural language questions with high accuracy. It leverages a **Retrieval-Augmented Generation (RAG)** architecture to provide contextually-aware and explainable answers, making it ideal for domains like insurance, legal, HR, and compliance.

-----

## ✨ Features

- **Multi-Format Document Processing:** Handles PDF documents fetched directly from a URL.
- **Semantic Search:** Uses Mistral's embedding API for fast and accurate clause retrieval.
- **LLM-Powered Decision Making:** Integrates with the **Mistral** LLM to generate precise answers based on retrieved document context.
- **API Key Rotation System:** Intelligent rotation across 10 Mistral API keys to prevent rate limiting during heavy testing.
- **Ultra-Fast Key Switching:** Optimized rotation system adds <1ms latency per request.
- **Automatic Failover:** Seamlessly switches to backup keys when rate limits are hit.
- **Explainable AI:** Answers are grounded in the provided text, ensuring traceability and minimizing hallucinations.
- **Structured Output:** Delivers responses in a clean, predictable JSON format.
- **Secure API:** The endpoint is protected via Bearer Token authentication.
- **Performance Monitoring:** Built-in endpoint to monitor API key usage and performance.

-----

## 🛠️ Tech Stack

- **Backend Framework:** FastAPI
- **Large Language Model (LLM):** Mistral (`mistral-large-latest`)
- **Embedding Model:** Mistral Embedding API (`mistral-embed`)
- **Vector Search:** Scikit-learn cosine similarity
- **Document Parsing:** PyPDF
- **API Key Management:** Custom rotation system with 10-key pool
- **Public Deployment:** ngrok for HTTPS tunneling

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

The application requires multiple Mistral API keys for optimal performance during hackathon testing.

- Create a `.env` file in the project root.
- Add your Mistral API keys (you can use 1-10 keys for rotation):

```env
# Primary API Key
MISTRAL_API_KEY="your_first_mistral_api_key_here"

# Additional API Keys for rotation (optional but recommended)
MISTRAL_API_KEY_1="your_first_mistral_api_key_here"
MISTRAL_API_KEY_2="your_second_mistral_api_key_here"
MISTRAL_API_KEY_3="your_third_mistral_api_key_here"
# ... up to MISTRAL_API_KEY_10
```

**Note:** The system automatically detects and uses all available API keys. Using multiple keys prevents rate limiting during intensive testing.

-----

## ▶️ Running the Application

Once the setup is complete, you can start the API server using Uvicorn.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will now be live and accessible at `http://localhost:8000`.

-----

## 🌐 Making Your API Public with ngrok (For Hackathon Submission)

To submit your solution to the hackathon platform, you need a publicly accessible URL. ngrok is the easiest way to expose your local server to the internet.

### Step 1: Install ngrok

#### For Windows:
1. Go to [https://dashboard.ngrok.com/get-started/setup/windows](https://dashboard.ngrok.com/get-started/setup/windows)
2. Login with Google (or create an account)
3. Download the Windows ZIP file
4. Extract the `ngrok.exe` file to a folder
5. Add the folder to your system PATH, or navigate to the folder in terminal

#### For macOS/Linux:
```bash
# Install via package manager (recommended)
# macOS with Homebrew:
brew install ngrok/ngrok/ngrok

# Ubuntu/Debian:
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok
```

### Step 2: Authenticate ngrok

1. Get your authtoken from [https://dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken)
2. Set your authtoken:

```bash
ngrok config add-authtoken "your_ngrok_authtoken_here"
```

### Step 3: Start Your Local Server

First, make sure your FastAPI application is running:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Keep this terminal open and running.

### Step 4: Expose Your Local Server

Open a **new terminal** and run:

```bash
ngrok http 8000
```

You should see output like this:

```
ngrok                                                                                                                                                                                                                   
                                                                                                                                                                                                                        
Session Status                online                                                                                                                                                                                    
Account                       your-email@example.com (Plan: Free)                                                                                                                                                      
Version                       3.3.0                                                                                                                                                                                     
Region                        United States (us)                                                                                                                                                                        
Latency                       45ms                                                                                                                                                                                      
Web Interface                 http://127.0.0.1:4040                                                                                                                                                                    
Forwarding                    https://abc123def456.ngrok-free.app -> http://localhost:8000                                                                                                                            
                                                                                                                                                                                                                        
Connections                   ttl     opn     rt1     rt5     p50     p90                                                                                                                                               
                              0       0       0.00    0.00    0.00    0.00     
```

### Step 5: Get Your Public URL

Copy the **HTTPS URL** from the "Forwarding" line. In the example above, it would be:
```
https://abc123def456.ngrok-free.app
```

### Step 6: Test Your Public API

Test your public API endpoint:

```bash
curl -X POST "https://your-ngrok-url.ngrok-free.app/hackrx/run" \
-H "Content-Type: application/json" \
-H "Accept: application/json" \
-H "Authorization: Bearer 02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0" \
-d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": ["What is the grace period for premium payment?"]
}'
```

### Step 7: Submit to Hackathon Platform

Use your ngrok URL for hackathon submission:
- **Webhook URL:** `https://your-ngrok-url.ngrok-free.app/hackrx/run`
- **Description:** FastAPI + Mistral LLM + API Key Rotation for RAG-based document Q&A

### 🔧 ngrok Pro Tips:

1. **Keep ngrok running:** Don't close the ngrok terminal while testing
2. **Monitor requests:** Visit `http://127.0.0.1:4040` to see all incoming requests
3. **Free tier limitations:** ngrok free tier has some rate limits, but should be fine for hackathon testing
4. **Stable URLs:** Free tier gives you random URLs. For custom domains, consider upgrading
5. **Security:** ngrok URLs are public - only use for testing/hackathons

### 🚨 Important Notes:

- **Keep both terminals open:** Your FastAPI server AND ngrok
- **HTTPS required:** Most hackathon platforms require HTTPS (ngrok provides this automatically)
- **URL changes:** Free ngrok URLs change each time you restart ngrok
- **No local network needed:** Works from anywhere with internet connection

-----

## 🧪 Testing Your Setup

### Test API Key Rotation System

Run the included test script to verify your API keys and rotation system:

```bash
python test_api_rotation.py
```

This will:
- ✅ Test all API keys are loaded correctly
- ⚡ Measure key switching speed 
- 🔄 Demonstrate rotation across multiple calls
- 📊 Show usage statistics
- 🧪 Test actual API calls

### Benchmark Performance

Test the speed of your key rotation system:

```bash
python benchmark_rotation_speed.py
```

Expected results:
- Key switching: `<1ms` per switch
- Total overhead: `<0.3%` of response time

### Monitor API Usage

Check real-time API key statistics:

```bash
curl -H "Authorization: Bearer 02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0" \
http://localhost:8000/api/stats
```

Or via your ngrok URL:

```bash
curl -H "Authorization: Bearer 02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0" \
https://your-ngrok-url.ngrok-free.app/api/stats
```

-----

## 🧪 API Usage

You can interact with the API by sending a `POST` request to the `/hackrx/run` endpoint.

- **Endpoint:** `POST /hackrx/run`
- **Authentication:** `Authorization: Bearer 02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0`
- **Local URL:** `http://localhost:8000/hackrx/run`
- **Public URL (via ngrok):** `https://your-ngrok-url.ngrok-free.app/hackrx/run`

### Sample cURL Request (Local Testing)

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

### Sample cURL Request (Public/Hackathon Testing)

```bash
curl -X POST "https://your-ngrok-url.ngrok-free.app/hackrx/run" \
-H "Content-Type: application/json" \
-H "Accept: application/json" \
-H "Authorization: Bearer 02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0" \
-d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for cataract surgery?"
    ]
}'se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
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