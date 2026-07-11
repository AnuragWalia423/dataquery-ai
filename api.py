# api.py  — FastAPI backend for DataQuery AI
# Ultra-simplified version - only essential endpoints with no numpy issues

import io
import os
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Annotated, Optional, List, Dict, Any
import json

import pandas as pd
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from auth import (
    init_auth_db, create_user, authenticate_user,
    create_access_token, decode_token,
    scoped_table_name, strip_user_prefix, filter_user_tables,
    user_has_uploaded_filename, register_uploaded_dataset,
)
from models import UserCreate, UserLogin, UserPublic, TokenData
from database_connection import DatabaseConnection
from sql_agent import EnhancedSQLAgent
from config import Config

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DataQuery AI",
    description="NL-to-SQL analytics API for finance and inventory datasets",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Boot user auth DB on startup
init_auth_db()

# ── Shared DB + Agent ────────────────────────────────────────────────────────
try:
    _db = DatabaseConnection(db_type=Config.DEFAULT_DATABASE or "sqlite")
    _agent = EnhancedSQLAgent(_db)
except Exception as e:
    logger.error(f"Failed to initialise database connection: {e}")
    _db = None
    _agent = None

# ── Auth helpers ──────────────────────────────────────────────────────────────
_bearer = HTTPBearer()

_AUTH_RATE_WINDOW_SECONDS = 60
_AUTH_RATE_MAX_ATTEMPTS = 10
_auth_attempts: Dict[str, deque] = defaultdict(deque)
_auth_rate_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_auth_rate_limit(request: Request, username: str, action: str) -> None:
    """Small in-process throttle for auth endpoints.

    This blocks obvious password-guessing loops on a single app process. For
    multi-worker or multi-instance production deployments, replace this with a
    shared store limiter such as slowapi + Redis.
    """
    now = time.monotonic()
    key = f"{action}:{_client_ip(request)}:{username.strip().lower()}"
    with _auth_rate_lock:
        attempts = _auth_attempts[key]
        while attempts and now - attempts[0] > _AUTH_RATE_WINDOW_SECONDS:
            attempts.popleft()
        if len(attempts) >= _AUTH_RATE_MAX_ATTEMPTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many authentication attempts. Please wait a minute and try again.",
            )
        attempts.append(now)


def get_current_user(
        credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)]
) -> TokenData:
    token_data = decode_token(credentials.credentials)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token_data


def get_agent() -> EnhancedSQLAgent:
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is not available. Check your .env configuration.",
        )
    return _agent


# ── Per-user active dataset tracking ─────────────────────────────────────────
# `_agent` above is ONE process-wide instance shared by every user and every
# request. Its `current_dataset` attribute is a single mutable field, so if we
# ever trust it as-is, one user's (or one browser tab's) request can silently
# flip the active dataset for every other in-flight request. To make dataset
# selection safe across concurrent users, we track each user's chosen dataset
# separately here and re-affirm it on the shared agent — under a lock — at the
# start of every request that reads or depends on `agent.current_dataset`.
_agent_lock = threading.Lock()
_user_active_dataset: Dict[str, str] = {}


def get_scoped_agent(
        current_user: Annotated[TokenData, Depends(get_current_user)],
        agent: Annotated[EnhancedSQLAgent, Depends(get_agent)],
) -> EnhancedSQLAgent:
    """Return the shared agent re-pointed at THIS user's active dataset.

    Must be used (instead of get_agent) by any endpoint that reads
    agent.current_dataset, to avoid acting on a dataset left active by a
    different user's request.
    """
    with _agent_lock:
        remembered = _user_active_dataset.get(current_user.user_id)
        if remembered:
            agent.current_dataset = remembered
    return agent


def _set_user_dataset(current_user: TokenData, agent: EnhancedSQLAgent, table_name: str) -> dict:
    """Switch the active dataset for THIS user only, recording it so it can be
    restored on the shared agent even after other users' requests run."""
    with _agent_lock:
        schema = agent.set_current_dataset(table_name)
        if "error" not in schema:
            _user_active_dataset[current_user.user_id] = table_name
        return schema


def get_db() -> DatabaseConnection:
    if _db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is not available.",
        )
    return _db


# ── Simple safe converter ──────────────────────────────────────────────────
def safe_convert(obj):
    """Safely convert any value to JSON-serializable format."""
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: safe_convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_convert(v) for v in obj]
    try:
        # Try to convert numpy/pandas types
        if hasattr(obj, 'item'):
            return safe_convert(obj.item())
        if hasattr(obj, 'tolist'):
            return safe_convert(obj.tolist())
        if hasattr(obj, 'to_dict'):
            return safe_convert(obj.to_dict())
        return str(obj)
    except:
        return str(obj)


def _safe_float(value, default=0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


# ── Request / Response models ─────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class SignupRequest(BaseModel):
    username: str
    email: str
    password: str
    confirm_password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class QueryRequest(BaseModel):
    question: str
    table_name: Optional[str] = None


class SetActiveDatasetRequest(BaseModel):
    table_name: str


class InventoryAnalysisRequest(BaseModel):
    analysis_type: str
    criteria: Optional[str] = None


# ── Auth endpoints ─────────────────────────────────────────────────────────────
@app.post("/api/auth/signup", response_model=TokenResponse)
def signup(req: SignupRequest, request: Request):
    _check_auth_rate_limit(request, req.username, "signup")
    try:
        payload = UserCreate(
            username=req.username,
            email=req.email,
            password=req.password,
            confirm_password=req.confirm_password,
        )
        user = create_user(payload)
        token = create_access_token(user)
        return TokenResponse(
            access_token=token,
            user={"user_id": user.user_id, "username": user.username, "email": user.email},
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/api/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, request: Request):
    _check_auth_rate_limit(request, req.username, "login")
    try:
        payload = UserLogin(username=req.username, password=req.password)
        user = authenticate_user(payload)
        if user is None:
            raise HTTPException(status_code=401, detail="Incorrect username or password.")
        token = create_access_token(user)
        return TokenResponse(
            access_token=token,
            user={"user_id": user.user_id, "username": user.username, "email": user.email},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/api/auth/me")
def me(current_user: Annotated[TokenData, Depends(get_current_user)]):
    return {"user_id": current_user.user_id, "username": current_user.username}


# ── Dataset / table endpoints ─────────────────────────────────────────────────
@app.get("/api/tables")
def list_tables(
        current_user: Annotated[TokenData, Depends(get_current_user)],
        agent: Annotated[EnhancedSQLAgent, Depends(get_scoped_agent)],
):
    """Return only the tables that belong to the logged-in user (display names)."""
    try:
        all_tables = agent._get_available_tables()
        user_tables = filter_user_tables(current_user.user_id, all_tables)
        display = [strip_user_prefix(current_user.user_id, t) for t in user_tables]
        current = strip_user_prefix(current_user.user_id, agent.current_dataset or "")
        return {
            "tables": display,
            "active": current if current in display else (display[0] if display else None),
        }
    except Exception as e:
        logger.error(f"List tables error: {e}")
        return {"tables": [], "active": None}


@app.post("/api/tables/active")
def set_active_table(
        req: SetActiveDatasetRequest,
        current_user: Annotated[TokenData, Depends(get_current_user)],
        agent: Annotated[EnhancedSQLAgent, Depends(get_agent)],
):
    actual = scoped_table_name(current_user.user_id, req.table_name)
    schema = _set_user_dataset(current_user, agent, actual)
    if "error" in schema:
        raise HTTPException(status_code=404, detail=f"Table '{req.table_name}' not found.")
    return {"active": req.table_name}


@app.get("/api/tables/check/{table_name}")
def check_table_exists(
        table_name: str,
        current_user: Annotated[TokenData, Depends(get_current_user)],
        agent: Annotated[EnhancedSQLAgent, Depends(get_agent)],
):
    """Check if a table already exists for the current user."""
    actual = scoped_table_name(current_user.user_id, table_name)
    all_tables = agent._get_available_tables()
    exists = actual in all_tables
    return {
        "exists": exists,
        "table_name": table_name,
        "message": f"Table '{table_name}' {'already exists' if exists else 'is available'}"
    }


# ── Upload endpoint ───────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_file(
        file: Annotated[UploadFile, File()],
        table_name: Annotated[str, Form()],
        current_user: Annotated[TokenData, Depends(get_current_user)],
        agent: Annotated[EnhancedSQLAgent, Depends(get_agent)],
):
    """Upload a file and load it into the user's isolated table."""
    contents = await file.read()
    ext = file.filename.rsplit(".", 1)[-1].lower()

    original_filename = file.filename.rsplit(".", 1)[0].lower()

    actual_table = scoped_table_name(current_user.user_id, table_name)
    all_tables = agent._get_available_tables()

    # Check 1: By table name
    if actual_table in all_tables:
        raise HTTPException(
            status_code=409,
            detail=f"Dataset '{table_name}' already exists. Please use a different name or delete the existing one."
        )

    # Check 2: By filename
    user_tables = filter_user_tables(current_user.user_id, all_tables)
    if user_has_uploaded_filename(current_user.user_id, file.filename):
        raise HTTPException(
            status_code=409,
            detail=f"A dataset with the same filename '{original_filename}' already exists. Please delete the existing one or rename the file."
        )

    for existing_table in user_tables:
        existing_display = strip_user_prefix(current_user.user_id, existing_table)
        if existing_display.lower() == original_filename:
            raise HTTPException(
                status_code=409,
                detail=f"A dataset with the same filename '{original_filename}' already exists. Please delete the existing one or rename the file."
            )

    try:
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(contents))
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(contents))
        elif ext == "json":
            df = pd.read_json(io.BytesIO(contents))
        elif ext == "parquet":
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    original_rows = len(df)

    # Apply advanced cleaning automatically
    try:
        from schemas_detector import SchemaDetector
        from database_normalisation import DataNormalizer
        schema_detector = SchemaDetector()
        data_normalizer = DataNormalizer()
        schema = schema_detector.detect_schema(df)
        cleaned_df = data_normalizer.normalize_dataframe(df, schema, level='advanced')
        cleaning_report = data_normalizer.get_last_report()
        cleaned_rows = len(cleaned_df)
        columns_added = len(cleaned_df.columns) - len(df.columns)
    except Exception as e:
        logger.warning(f"Cleaning failed, using raw data: {e}")
        cleaned_df = df
        cleaning_report = {"error": str(e), "used_raw_data": True}
        cleaned_rows = len(df)
        columns_added = 0

    success = agent.load_dataset(cleaned_df, actual_table, already_normalized=True)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to load dataset into database.")

    try:
        register_uploaded_dataset(current_user.user_id, file.filename, actual_table)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    with _agent_lock:
        agent.detect_schema(actual_table)
        # A freshly uploaded table becomes this user's active dataset. Recording
        # it here (not just relying on whatever detect_schema set on the shared
        # agent) is what lets get_scoped_agent restore it correctly later, even
        # after other users' requests have pointed the shared agent elsewhere.
        _user_active_dataset[current_user.user_id] = actual_table

    return {
        "table": table_name,
        "rows": len(cleaned_df),
        "columns": list(cleaned_df.columns),
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "columns_added": columns_added,
        "cleaning_report": safe_convert(cleaning_report),
        "filename": original_filename,
        "message": f"Loaded {cleaned_rows} rows from '{file.filename}' as '{table_name}'"
    }


# ── Cleaning endpoint ─────────────────────────────────────────────────────────
@app.post("/api/clean")
async def clean_data(
        file: Annotated[UploadFile, File()],
        cleaning_level: Annotated[str, Form()] = "advanced",
        current_user: Annotated[TokenData, Depends(get_current_user)] = None,
):
    """Clean a dataset with the specified cleaning level."""
    contents = await file.read()
    ext = file.filename.rsplit(".", 1)[-1].lower()

    try:
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(contents))
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(io.BytesIO(contents))
        elif ext == "json":
            df = pd.read_json(io.BytesIO(contents))
        elif ext == "parquet":
            df = pd.read_parquet(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    original_rows = len(df)

    try:
        from schemas_detector import SchemaDetector
        from database_normalisation import DataNormalizer
        schema_detector = SchemaDetector()
        data_normalizer = DataNormalizer()
        schema = schema_detector.detect_schema(df)

        valid_levels = {"basic", "standard", "advanced"}
        if cleaning_level not in valid_levels:
            cleaning_level = "advanced"

        cleaned_df = data_normalizer.normalize_dataframe(df, schema, level=cleaning_level)
        cleaning_report = data_normalizer.get_last_report()
        cleaned_rows = len(cleaned_df)
        columns_added = len(cleaned_df.columns) - len(df.columns)

        clean_data = cleaned_df.head(100).replace({np.nan: None}).to_dict(orient="records")
        clean_data = safe_convert(clean_data)

        return {
            "original_rows": original_rows,
            "cleaned_rows": cleaned_rows,
            "columns_added": columns_added,
            "cleaning_report": safe_convert(cleaning_report),
            "cleaned_data": clean_data,
            "columns": list(cleaned_df.columns),
            "message": f"Cleaning complete ({cleaning_level}): {original_rows} → {cleaned_rows} rows"
        }
    except Exception as e:
        logger.error(f"Cleaning error: {e}")
        raise HTTPException(status_code=500, detail=f"Cleaning failed: {str(e)}")


# ── Query endpoint ────────────────────────────────────────────────────────────
@app.post("/api/query")
def run_query(
        req: QueryRequest,
        current_user: Annotated[TokenData, Depends(get_current_user)],
        agent: Annotated[EnhancedSQLAgent, Depends(get_scoped_agent)],
):
    """Translate a natural-language question to SQL and return the result + insight."""
    if req.table_name:
        actual = scoped_table_name(current_user.user_id, req.table_name)
        _set_user_dataset(current_user, agent, actual)

    if not agent.current_dataset:
        raise HTTPException(status_code=400, detail="No active dataset. Upload a file first.")

    try:
        with _agent_lock:
            # Re-affirm right before querying: another user's request could have
            # run (and changed agent.current_dataset) between the dependency
            # resolving above and this line.
            agent.current_dataset = _user_active_dataset.get(current_user.user_id, agent.current_dataset)
            response = agent.answer_question(req.question)
            dataset_used = agent.current_dataset
        return {
            "question": req.question,
            "response": response,
            "dataset": strip_user_prefix(current_user.user_id, dataset_used),
            "provider": getattr(agent.llm_manager, "last_provider", "unknown"),
        }
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── KPI endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/kpi/all")
def get_all_kpis(
        current_user: Annotated[TokenData, Depends(get_current_user)],
        agent: Annotated[EnhancedSQLAgent, Depends(get_scoped_agent)],
):
    """Get all KPI values in a single request for faster loading."""
    if not agent.current_dataset:
        return {
            "revenue": 0,
            "profit": 0,
            "orders": 0,
            "customers": 0,
            "error": "No active dataset"
        }

    try:
        with _agent_lock:
            # Re-affirm right before use: closes the race window between the
            # dependency resolving above and the queries below, during which
            # another user's request could have swapped the shared dataset.
            agent.current_dataset = _user_active_dataset.get(current_user.user_id, agent.current_dataset)
            table = agent.current_dataset
            schema = agent.detect_schema()
            mapping = schema.get('column_mapping', {})

            kpi_data = {"revenue": 0, "profit": 0, "orders": 0, "customers": 0}

            queries = {}

            if mapping.get('sales'):
                queries['revenue'] = f"SELECT ROUND(SUM({mapping['sales']}), 2) as revenue FROM {table}"

            if mapping.get('profit'):
                queries['profit'] = f"SELECT ROUND(SUM({mapping['profit']}), 2) as profit FROM {table}"

            if mapping.get('order_id'):
                queries['orders'] = f"SELECT COUNT(DISTINCT {mapping['order_id']}) as orders FROM {table}"
            elif mapping.get('transaction_id'):
                queries['orders'] = f"SELECT COUNT(DISTINCT {mapping['transaction_id']}) as orders FROM {table}"

            if mapping.get('customer_id'):
                queries['customers'] = f"SELECT COUNT(DISTINCT {mapping['customer_id']}) as customers FROM {table}"

            for key, query in queries.items():
                try:
                    result = _db.execute_query(query)
                    if not result.empty:
                        val = result.iloc[0, 0]
                        if val is not None and not pd.isna(val):
                            if isinstance(val, (np.integer, np.int64)):
                                kpi_data[key] = int(val)
                            elif isinstance(val, (np.floating, np.float64)):
                                kpi_data[key] = float(val)
                            else:
                                kpi_data[key] = float(val) if isinstance(val, (int, float)) else val
                except Exception as e:
                    logger.warning(f"KPI {key} query failed: {e}")

        return {
            "revenue": float(kpi_data["revenue"]) if kpi_data["revenue"] else 0,
            "profit": float(kpi_data["profit"]) if kpi_data["profit"] else 0,
            "orders": int(kpi_data["orders"]) if kpi_data["orders"] else 0,
            "customers": int(kpi_data["customers"]) if kpi_data["customers"] else 0,
        }

    except Exception as e:
        logger.error(f"KPI error: {e}")
        return {
            "revenue": 0,
            "profit": 0,
            "orders": 0,
            "customers": 0,
            "error": str(e)
        }


# ── Inventory endpoints ──────────────────────────────────────────────────────
@app.get("/api/inventory/metrics")
def get_inventory_metrics(
        current_user: Annotated[TokenData, Depends(get_current_user)],
        agent: Annotated[EnhancedSQLAgent, Depends(get_scoped_agent)],
):
    """Get inventory-specific metrics for the inventory analytics page."""
    if not agent.current_dataset:
        return {
            "total_stock": 0,
            "inventory_value": 0,
            "low_stock": 0,
            "out_of_stock": 0,
            "error": "No active dataset"
        }

    try:
        with _agent_lock:
            agent.current_dataset = _user_active_dataset.get(current_user.user_id, agent.current_dataset)
            table = agent.current_dataset
            schema = agent.detect_schema()
            inventory_cols = schema.get('inventory_columns', {})

            metrics = {"total_stock": 0, "inventory_value": 0, "low_stock": 0, "out_of_stock": 0}

            stock_col = inventory_cols.get('stock_level')
            reorder_col = inventory_cols.get('reorder_point')

            if stock_col:
                try:
                    result = _db.execute_query(f"SELECT SUM({stock_col}) as total FROM {table}")
                    if not result.empty:
                        val = result.iloc[0, 0]
                        metrics['total_stock'] = _safe_float(val)
                except Exception as e:
                    logger.warning(f"Total stock query failed: {e}")

                try:
                    result = _db.execute_query(
                        f"SELECT COUNT(*) as count FROM {table} WHERE {stock_col} = 0 OR {stock_col} IS NULL")
                    if not result.empty:
                        val = result.iloc[0, 0]
                        metrics['out_of_stock'] = _safe_int(val)
                except Exception as e:
                    logger.warning(f"Out of stock query failed: {e}")

                if reorder_col:
                    try:
                        result = _db.execute_query(
                            f"SELECT COUNT(*) as count FROM {table} WHERE {stock_col} <= {reorder_col} AND {stock_col} > 0")
                        if not result.empty:
                            val = result.iloc[0, 0]
                            metrics['low_stock'] = _safe_int(val)
                    except Exception as e:
                        logger.warning(f"Low stock query failed: {e}")

                cost_col = None
                for col in schema.get('numeric_columns', []):
                    if any(term in col.lower() for term in ['cost', 'price', 'unit_cost']):
                        cost_col = col
                        break

                if cost_col and cost_col != stock_col:
                    try:
                        result = _db.execute_query(f"SELECT SUM({stock_col} * {cost_col}) as value FROM {table}")
                        if not result.empty:
                            val = result.iloc[0, 0]
                            metrics['inventory_value'] = _safe_float(val)
                    except Exception as e:
                        logger.warning(f"Inventory value query failed: {e}")

        return {
            "total_stock": float(metrics["total_stock"]),
            "inventory_value": float(metrics["inventory_value"]),
            "low_stock": int(metrics["low_stock"]),
            "out_of_stock": int(metrics["out_of_stock"]),
        }

    except Exception as e:
        logger.error(f"Inventory metrics error: {e}")
        return {
            "total_stock": 0,
            "inventory_value": 0,
            "low_stock": 0,
            "out_of_stock": 0,
            "error": str(e)
        }


# The frontend calls this in two places: the sidebar's "Type: X" dataset
# badge, and the Inventory Analytics page's checkInventoryDataset(), which
# gates the whole page on schema.dataset_type === 'inventory_management'.
# Neither caller existed as a registered route until now -- both had a
# `if (!res.ok) return;` guard, so a 404 here made them silently do nothing,
# which is why the Inventory Analytics warning banner never went away even
# for a genuine inventory dataset.
@app.get("/api/schema")
def get_schema(
        current_user: Annotated[TokenData, Depends(get_current_user)],
        agent: Annotated[EnhancedSQLAgent, Depends(get_scoped_agent)],
):
    """Return the active dataset's detected schema."""
    if not agent.current_dataset:
        return {"error": "No active dataset", "dataset_type": None}

    with _agent_lock:
        agent.current_dataset = _user_active_dataset.get(current_user.user_id, agent.current_dataset)
        schema = agent.detect_schema()

    if "error" in schema:
        raise HTTPException(status_code=400, detail=schema["error"])

    # Returned as an explicit, plain-Python-typed subset rather than the raw
    # schema dict -- the full dict's per-column 'sample_values' can contain
    # numpy scalar types that aren't guaranteed to survive default JSON
    # serialization, and the frontend only ever reads a handful of these
    # fields anyway.
    row_count = schema.get("row_count")
    return {
        "table_name": schema.get("table_name"),
        "dataset_type": schema.get("dataset_type"),
        "column_mapping": dict(schema.get("column_mapping", {})),
        "inventory_columns": dict(schema.get("inventory_columns", {})),
        "date_columns": list(schema.get("date_columns", [])),
        "numeric_columns": list(schema.get("numeric_columns", [])),
        "string_columns": list(schema.get("string_columns", [])),
        "row_count": int(row_count) if row_count is not None else None,
    }


# The 8 options in the Inventory Analytics dropdown map straight to these
# named analyses. Both agent.inventory_analysis() and agent.abc_analysis()
# already existed as hand-written, tested SQL builders, but nothing ever
# called them -- the dropdown used to send natural-language text through
# /api/query and let the LLM guess at inventory SQL from scratch every time.
# This endpoint wires the dropdown directly to those methods instead.
_INVENTORY_ANALYSIS_TYPES = {
    "stock_status", "inventory_value", "turnover_analysis", "reorder_analysis",
    "supplier_analysis", "warehouse_analysis", "expiry_analysis", "abc_classification",
}


@app.post("/api/inventory/analysis")
def run_inventory_analysis(
        req: InventoryAnalysisRequest,
        current_user: Annotated[TokenData, Depends(get_current_user)],
        agent: Annotated[EnhancedSQLAgent, Depends(get_scoped_agent)],
):
    """Run a predefined inventory analysis directly, bypassing the LLM."""
    if req.analysis_type not in _INVENTORY_ANALYSIS_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown analysis_type '{req.analysis_type}'. Valid options: {sorted(_INVENTORY_ANALYSIS_TYPES)}",
        )

    if not agent.current_dataset:
        raise HTTPException(status_code=400, detail="No active dataset. Upload a file first.")

    try:
        with _agent_lock:
            # Re-affirm right before use, same as every other dataset-reading
            # endpoint -- closes the race window against other users' requests.
            agent.current_dataset = _user_active_dataset.get(current_user.user_id, agent.current_dataset)
            if req.analysis_type == "abc_classification":
                df = agent.abc_analysis(criteria=req.criteria or "value")
            else:
                df = agent.inventory_analysis(req.analysis_type)
    except Exception as e:
        logger.error(f"Inventory analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if df is None or df.empty:
        return {"columns": [], "rows": [], "message": "No data found"}

    # inventory_analysis()/abc_analysis() return a single-column 'message'
    # DataFrame (not an error/exception) when there's a known reason for no
    # results -- e.g. no inventory columns detected, or an unsupported
    # analysis type for this particular dataset's columns.
    if list(df.columns) == ["message"]:
        return {"columns": [], "rows": [], "message": str(df.iloc[0, 0])}

    safe_df = df.astype(object).where(pd.notnull(df), None)
    return {"columns": df.columns.tolist(), "rows": safe_df.values.tolist()}


# ── Data Explorer endpoint ────────────────────────────────────────────────────
@app.post("/api/explorer")
async def run_explorer_query(
        request: Request,
        current_user: Annotated[TokenData, Depends(get_current_user)],
        agent: Annotated[EnhancedSQLAgent, Depends(get_scoped_agent)],
):
    """Execute a custom SQL query for the Data Explorer page."""
    try:
        body = await request.json()
        query = body.get('query', '')
    except:
        raise HTTPException(status_code=400, detail="Invalid request body. Please send {'query': 'your SQL'}.")

    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not agent.current_dataset:
        raise HTTPException(status_code=400, detail="No active dataset. Please load a dataset first.")

    try:
        result = _db.execute_query(query)
        if result.empty:
            return {
                "rows": [],
                "columns": [],
                "count": 0,
                "message": "Query returned no results."
            }

        data = result.replace({np.nan: None}).to_dict(orient="records")
        data = safe_convert(data)
        columns = list(result.columns)

        return {
            "rows": data,
            "columns": columns,
            "count": len(result),
            "message": f"✅ Query returned {len(result)} rows."
        }
    except Exception as e:
        logger.error(f"Explorer query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


# ── System status endpoint ────────────────────────────────────────────────────
@app.get("/api/status")
def system_status():
    """Health check — no auth required so the frontend can show live status."""
    db_ok = _db is not None
    return {
        "database": "connected" if db_ok else "disconnected",
        "db_type": getattr(_db, "db_type", "unknown") if db_ok else "none",
        "gemini": "configured" if Config.GEMINI_API_KEY else "missing",
        "groq": "configured" if getattr(Config, "GROQ_API_KEY", None) else "missing",
    }


# ── Serve frontend ────────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


    @app.get("/", include_in_schema=False)
    def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
