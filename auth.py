# auth.py
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from dotenv import load_dotenv
from passlib.context import CryptContext
from sqlalchemy import (
    create_engine, text, Column, Integer, String, Boolean,
    UniqueConstraint, event
)
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.exc import IntegrityError

from config import Config
from models import UserCreate, UserLogin, UserInDB, UserPublic, TokenData

logger = logging.getLogger(__name__)
load_dotenv()
load_dotenv("env")

# ── Config ────────────────────────────────────────────────────────────────────
# JWT_SECRET_KEY must be supplied by the deployment environment. A hardcoded
# fallback would let anyone who has seen the source forge valid JWTs.
try:
    SECRET_KEY = os.environ["JWT_SECRET_KEY"]
except KeyError as e:
    raise RuntimeError(
        "JWT_SECRET_KEY environment variable is not set. Generate one with "
        "`python -c \"import secrets; print(secrets.token_urlsafe(48))\"` "
        "and set it in your environment, .env file, or env file before starting the app."
    ) from e
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24
AUTH_DB_PATH = os.environ.get("AUTH_DB_PATH", "auth_users.db")

# Escape hatch for deployments that want the auth DB on a *different*
# Postgres instance than the one holding uploaded datasets (rare — most
# deployments should just set DEFAULT_DATABASE=postgresql + POSTGRES_* and
# let both use the same instance, see _make_engine() below). Accepts a full
# SQLAlchemy URL, e.g. postgresql+psycopg2://user:pass@host:5432/dbname
AUTH_DATABASE_URL = (
    os.environ.get("AUTH_DATABASE_URL", "").strip()
    or os.environ.get("DATABASE_URL", "").strip()
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── SQLAlchemy setup ──────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class UserModel(Base):
    """ORM model for the users table."""
    __tablename__ = "users"

    user_id         = Column(Integer, primary_key=True, autoincrement=True)
    username        = Column(String(30), nullable=False, unique=True)
    email           = Column(String(254), nullable=False, unique=True)
    hashed_password = Column(String(128), nullable=False)
    is_active       = Column(Boolean, nullable=False, default=True)
    created_at      = Column(String(32), nullable=False,
                             default=lambda: datetime.now(timezone.utc).isoformat())

    __table_args__ = (
        UniqueConstraint("username", name="uq_users_username"),
        UniqueConstraint("email",    name="uq_users_email"),
    )


class UploadedDatasetModel(Base):
    """Track uploaded source filenames per user to prevent duplicate datasets."""
    __tablename__ = "uploaded_datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, nullable=False)
    filename_key = Column(String(255), nullable=False)
    table_name = Column(String(255), nullable=False)
    created_at = Column(String(32), nullable=False,
                        default=lambda: datetime.now(timezone.utc).isoformat())

    __table_args__ = (
        UniqueConstraint("user_id", "filename_key", name="uq_user_dataset_filename"),
        UniqueConstraint("user_id", "table_name", name="uq_user_dataset_table"),
    )


def _postgres_url_from_config() -> URL:
    """Build a postgres+psycopg2 URL from the same POSTGRES_* settings that
    database_connection.py already uses for uploaded-dataset storage, so
    flipping DEFAULT_DATABASE=postgresql moves auth + data together without
    needing a second set of env vars."""
    return URL.create(
        "postgresql+psycopg2",
        username=Config.POSTGRES_USER,
        password=Config.POSTGRES_PASSWORD,
        host=Config.POSTGRES_HOST,
        port=int(Config.POSTGRES_PORT),
        database=Config.POSTGRES_DATABASE,
    )


def _make_engine():
    """Build the auth-DB engine.

    Resolution order (first match wins):
      1. AUTH_DATABASE_URL — explicit override for split deployments where
         auth lives on a different Postgres instance than uploaded data.
      2. DEFAULT_DATABASE=postgresql/postgres — reuse the same POSTGRES_*
         settings database_connection.py uses, so one env-var switch moves
         both the auth DB and the uploaded-dataset DB to Postgres together.
      3. SQLite fallback (AUTH_DB_PATH) — local/dev default. NOT durable on
         most hosting platforms' ephemeral filesystems; do not use this in
         production.
    """
    if AUTH_DATABASE_URL:
        url = AUTH_DATABASE_URL
        # Some platforms (Render/Heroku/Railway) inject the legacy
        # "postgres://" scheme, which SQLAlchemy 2.x rejects outright.
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return create_engine(url, pool_pre_ping=True, pool_recycle=1800)

    default_db = (Config.DEFAULT_DATABASE or "sqlite").lower()
    if default_db in ("postgres", "postgresql"):
        try:
            import psycopg2  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "psycopg2-binary is required to run the auth DB on Postgres. "
                "Install it with: pip install psycopg2-binary"
            ) from e
        return create_engine(
            _postgres_url_from_config(),
            pool_pre_ping=True,
            pool_recycle=1800,
        )

    # ── SQLite fallback (dev only) ──────────────────────────────────────
    engine = create_engine(
        f"sqlite:///{AUTH_DB_PATH}",
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
    # Enforce foreign-key support and WAL mode for better concurrency
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    return engine


# Module-level engine + session factory — created once, reused everywhere.
# This is the same pattern DatabaseConnection uses and plays nicely with
# SQLAlchemy's own connection pool rather than fighting it.
_engine = _make_engine()
_SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)


def init_auth_db() -> None:
    """Create the users table if it doesn't exist. Safe to call on every startup."""
    Base.metadata.create_all(_engine)


def _session() -> Session:
    """Return a new SQLAlchemy Session from the module-level factory."""
    return _SessionFactory()


# ── Password helpers ──────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ── User CRUD ─────────────────────────────────────────────────────────────────
def create_user(payload: UserCreate) -> UserPublic:
    """Insert a new user. Raises ValueError on duplicate username/email."""
    hashed = hash_password(payload.password)
    row = UserModel(
        username=payload.username.lower(),
        email=payload.email.lower(),
        hashed_password=hashed,
    )
    with _session() as session:
        try:
            session.add(row)
            session.commit()
            session.refresh(row)
            return UserPublic(
                user_id=row.user_id,
                username=row.username,
                email=row.email,
            )
        except IntegrityError as e:
            session.rollback()
            msg = str(e.orig).lower() if e.orig else str(e).lower()
            if "username" in msg:
                raise ValueError("Username already taken. Please choose another.")
            if "email" in msg:
                raise ValueError("An account with that email already exists.")
            raise ValueError("Registration failed — please try again.")


def get_user_by_username(username: str) -> Optional[UserInDB]:
    with _session() as session:
        row = session.execute(
            text("SELECT * FROM users WHERE LOWER(username) = LOWER(:u)"),
            {"u": username.strip()}
        ).mappings().fetchone()

    if not row:
        return None
    return UserInDB(
        user_id=row["user_id"],
        username=row["username"],
        email=row["email"],
        hashed_password=row["hashed_password"],
        is_active=bool(row["is_active"]),
    )


def authenticate_user(payload: UserLogin) -> Optional[UserPublic]:
    """Return UserPublic if credentials are valid, None otherwise."""
    user = get_user_by_username(payload.username)
    if not user or not user.is_active:
        return None
    if not verify_password(payload.password, user.hashed_password):
        return None
    return UserPublic(user_id=user.user_id, username=user.username, email=user.email)


# ── JWT ───────────────────────────────────────────────────────────────────────
def create_access_token(user: UserPublic) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    return jwt.encode(
        {
            "sub":      str(user.user_id),
            "username": user.username,
            "exp":      expire,
            "iat":      datetime.now(timezone.utc),
        },
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def decode_token(token: str) -> Optional[TokenData]:
    """Decode and validate a JWT. Returns None if invalid or expired."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return TokenData(user_id=int(payload["sub"]), username=payload["username"])
    except jwt.ExpiredSignatureError:
        logger.info("JWT expired — user must log in again.")
        return None
    except (jwt.InvalidTokenError, KeyError, ValueError) as e:
        logger.warning(f"Invalid JWT: {e}")
        return None


# ── Table isolation helpers ───────────────────────────────────────────────────
def user_table_prefix(user_id: int) -> str:
    """Namespace prefix for this user's tables: 'u3_'"""
    return f"u{user_id}_"


def scoped_table_name(user_id: int, base_name: str) -> str:
    """
    Prefix a user-chosen table name with their user ID so that tables from
    different users never collide inside the shared SQLite file.

    Examples
    --------
    user 1 uploads 'sales'     → stored as 'u1_sales'
    user 2 uploads 'sales'     → stored as 'u2_sales'   (different row)
    """
    prefix = user_table_prefix(user_id)
    return base_name if base_name.startswith(prefix) else f"{prefix}{base_name}"


def strip_user_prefix(user_id: int, table_name: str) -> str:
    """Remove the user prefix for display — 'u3_sales' → 'sales'."""
    prefix = user_table_prefix(user_id)
    return table_name[len(prefix):] if table_name.startswith(prefix) else table_name


def filter_user_tables(user_id: int, all_tables: list) -> list:
    """Return only the tables that belong to this user."""
    prefix = user_table_prefix(user_id)
    return [t for t in all_tables if t.startswith(prefix)]


def normalize_filename_key(filename: str) -> str:
    """Canonical key for duplicate uploaded-file detection."""
    return (filename or "").rsplit(".", 1)[0].strip().lower()


def user_has_uploaded_filename(user_id: int, filename: str) -> bool:
    filename_key = normalize_filename_key(filename)
    if not filename_key:
        return False
    with _session() as session:
        row = session.execute(
            text(
                "SELECT 1 FROM uploaded_datasets "
                "WHERE user_id = :user_id AND filename_key = :filename_key LIMIT 1"
            ),
            {"user_id": user_id, "filename_key": filename_key},
        ).fetchone()
        return row is not None


def register_uploaded_dataset(user_id: int, filename: str, table_name: str) -> None:
    filename_key = normalize_filename_key(filename)
    if not filename_key:
        return
    row = UploadedDatasetModel(
        user_id=user_id,
        filename_key=filename_key,
        table_name=table_name,
    )
    with _session() as session:
        try:
            session.add(row)
            session.commit()
        except IntegrityError:
            session.rollback()
            raise ValueError("This dataset has already been uploaded for this user.")
