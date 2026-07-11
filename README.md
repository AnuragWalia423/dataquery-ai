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
- Streamlit legacy/alternate UI

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

## Deployment Notes

For a free demo deployment, Render is a good fit.

Suggested Render start command:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

Set these environment variables in Render:

```env
JWT_SECRET_KEY=your_long_random_secret
GROQ_API_KEY=your_key_optional
GEMINI_API_KEY=your_key_optional
```

Note: local SQLite files are not durable on most free hosting platforms. For a production deployment, move user/auth and uploaded dataset storage to Postgres.

