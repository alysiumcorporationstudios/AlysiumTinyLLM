# Alysium TinyLM – API

Simple FastAPI server for the character-level TinyLM.

## Local test

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open:
- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs  (interactive docs)

### Example request

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hello"}'
```

## Deploy on Render (Recommended)

1. Create a new **Web Service** on Render
2. Connect your GitHub repository
3. Use these settings:

   - **Root Directory**: `api`   (important!)
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. Click Deploy

After it finishes you will get a URL like:
`https://alysium-tinylm.onrender.com`

### Important notes for Render free tier
- The service will sleep after some inactivity
- First request after sleep can take 30–60 seconds
- Perfect for personal / low-traffic use

## API Endpoints

| Method | Endpoint   | Description              |
|--------|------------|--------------------------|
| GET    | `/`        | Basic info               |
| GET    | `/health`  | Health check             |
| POST   | `/chat`    | Send message, get reply  |
| GET    | `/docs`    | Interactive documentation|

### POST /chat body

```json
{
  "message": "hello how are you",
  "max_new_chars": 80,
  "temperature": 0.7,
  "top_k": 15
}
```
