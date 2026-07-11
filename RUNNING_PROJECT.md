# Running This Project Locally

This project now runs primarily as a FastAPI backend serving the frontend from
`frontend/index.html`.

## 1. Use the Project Virtual Environment

From the project folder:

```powershell
cd C:\Users\anura\PycharmProjects\PythonProject19
.\.venv\Scripts\Activate.ps1
```

If the virtual environment does not exist yet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Configure Environment Variables

The app refuses to start without `JWT_SECRET_KEY`.

Generate one:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Add it to either `env` or `.env`:

```env
JWT_SECRET_KEY=your_generated_secret_here
GROQ_API_KEY=your_groq_key_optional
GEMINI_API_KEY=your_gemini_key_optional
```

The code loads both files:

- `.env`
- `env`

## 3. Run the FastAPI App

```powershell
uvicorn api:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## 4. Run Tests

```powershell
python -m pytest -q
```

The test suite covers:

- Superstore upload and KPI flow
- Inventory upload and inventory analytics flow
- ABC classification
- Generic data cleaning
- Per-user dataset isolation
- Same-user duplicate filename prevention

## 5. Deployment Start Command

For Render or another ASGI host:

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

Remember to set `JWT_SECRET_KEY` in the hosting platform's environment
variables.
