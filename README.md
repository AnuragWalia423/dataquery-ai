# DataQuery AI Dashboard

AI-powered data analytics dashboard for uploading datasets, cleaning them automatically, querying them in natural language, and exploring KPI/inventory insights.

## Features

- User authentication with JWT-based sessions
- Per-user dataset isolation
- Duplicate-upload protection per user
- Universal data cleaning pipeline for CSV, Excel, JSON, and Parquet uploads
- Automatic schema detection
- Natural-language-to-SQL analytics with Gemini/Groq support
- KPI dashboard for sales/order datasets
- Inventory analytics dashboard with:
  - stock status
  - inventory value
  - reorder analysis
  - supplier and warehouse analysis
  - expiry analysis
  - ABC classification
- PDF download support for KPI, dashboard, inventory, and LLM insight results
- FastAPI backend with static frontend

## Tech Stack

- Python
- FastAPI
- SQLAlchemy
- Pandas / NumPy
- LangChain
- Gemini / Groq LLM integrations
- Chart.js frontend
- Docker/Render deployment support

## Local Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create an `env` or `.env` file and add:

```env
JWT_SECRET_KEY=your_long_random_secret_here
GROQ_API_KEY=your_groq_key_optional
GEMINI_API_KEY=your_gemini_key_optional
```

Generate a JWT secret with:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Run the FastAPI app:

```powershell
uvicorn api:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Testing

Run:

```powershell
python -m pytest -q
```

Current smoke tests cover:

- Superstore upload and KPI flow
- Inventory upload and analytics flow
- ABC classification
- Data cleaning pipeline
- Per-user dataset visibility
- Duplicate dataset upload prevention

## Deployment on Render Free Tier

This repository includes a `Dockerfile`, `.dockerignore`, `requirements-render.txt`, and `render.yaml` blueprint for Render. The blueprint creates a free Render web service and a free Render Postgres database.

### Option A: Deploy from `render.yaml`

1. Push this repository to GitHub.
2. In Render, choose **New > Blueprint**.
3. Connect this repository.
4. Render will read `render.yaml` and create the `dataquery-ai` web service.
5. Enter `GROQ_API_KEY` and/or `GEMINI_API_KEY` when prompted, or leave them blank and add later.

### Option B: Manual Web Service

If you create a manual Render Web Service:

- Runtime: `Docker`
- Instance type: `Free`
- Branch: `main`
- Health check path: `/api/status`
- Dockerfile path: `./Dockerfile`

The Docker container starts the app with:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

Environment variables:

```env
JWT_SECRET_KEY=generate_or_set_a_long_random_secret
DEFAULT_DATABASE=postgresql
DATABASE_URL=provided_by_Render_Postgres
AUTH_DATABASE_URL=provided_by_Render_Postgres
UPLOAD_DIR=/data/uploads
GROQ_API_KEY=your_key_optional
GEMINI_API_KEY=your_key_optional
```

### Free Tier Limitations

Render Free web services spin down after inactivity. The app stores auth records and uploaded dataset tables in Render Postgres when deployed through `render.yaml`. Files written to the container filesystem remain ephemeral, so keep long-term file/blob storage outside the container if you add persistent file uploads later.
