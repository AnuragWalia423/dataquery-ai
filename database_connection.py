# database.py
import re
import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from config import Config
import pandas as pd


class SQLValidationError(Exception):
    """Raised when a query fails the forbidden-keyword / multi-statement safety check"""
    pass


class UnsupportedDatabaseError(Exception):
    """Raised when an unsupported db_type is requested"""
    pass


class DatabaseConnection:
    """Connection wrapper supporting MySQL, PostgreSQL, and SQLite.

    Previously this class was hardcoded to MySQL only, even though
    config.py defines connection settings for Postgres, MongoDB and SQLite
    and the app is titled "Universal SQL Database Agent". MongoDB is
    intentionally NOT supported here: it's a document store with no SQL
    dialect, and the rest of this project (query validation, the LLM's
    SQL-generation prompts, DESCRIBE/EXPLAIN-style schema inspection) is
    built entirely around relational/SQL semantics. Bolting on a
    MongoDB aggregation-pipeline code path would mean a second, mostly
    duplicate agent implementation rather than a small fix, so it's left
    as an explicit "not supported" case below instead of a half-working
    shim.
    """

    SUPPORTED_DATABASES = {'mysql', 'postgresql', 'sqlite'}

    def __init__(self, db_type: str = None):
        requested = (db_type or Config.DEFAULT_DATABASE or 'sqlite').lower()
        if requested in ('postgres', 'postgresql'):
            requested = 'postgresql'

        if requested == 'mongodb':
            raise UnsupportedDatabaseError(
                "MongoDB is not supported by this SQL-based agent. "
                "Supported databases: mysql, postgresql, sqlite."
            )
        if requested not in self.SUPPORTED_DATABASES:
            raise UnsupportedDatabaseError(
                f"Unsupported database type '{requested}'. "
                f"Supported databases: {', '.join(sorted(self.SUPPORTED_DATABASES))}."
            )

        self.db_type = requested

        if self.db_type == 'mysql':
            self.host = Config.MYSQL_HOST
            self.user = Config.MYSQL_USER
            self.password = Config.MYSQL_PASSWORD
            self.database = Config.MYSQL_DATABASE
            self.port = int(Config.MYSQL_PORT)
        elif self.db_type == 'postgresql':
            self.host = Config.POSTGRES_HOST
            self.user = Config.POSTGRES_USER
            self.password = Config.POSTGRES_PASSWORD
            self.database = Config.POSTGRES_DATABASE
            self.port = int(Config.POSTGRES_PORT)
            self.database_url = os.getenv("DATABASE_URL", "").strip()
        else:  # sqlite
            self.sqlite_path = Config.SQLITE_DATABASE

        # Create SQLAlchemy engine
        self.engine = self.create_engine()
        self.Session = sessionmaker(bind=self.engine)

    def create_engine(self):
        """Create SQLAlchemy engine for the configured backend"""
        if self.db_type == 'mysql':
            connection_url = URL.create(
                "mysql+pymysql",
                username=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database,
                query={"charset": "utf8mb4"}
            )
            return create_engine(
                connection_url,
                poolclass=NullPool,
                pool_pre_ping=True,
                pool_recycle=3600
            )
        elif self.db_type == 'postgresql':
            if self.database_url:
                # Render/Heroku-style URLs commonly use postgresql:// or the
                # older postgres:// scheme. SQLAlchemy 2 works reliably with
                # the explicit psycopg2 driver.
                url = self.database_url
                if url.startswith("postgres://"):
                    url = url.replace("postgres://", "postgresql+psycopg2://", 1)
                elif url.startswith("postgresql://"):
                    url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
                return create_engine(
                    url,
                    poolclass=NullPool,
                    pool_pre_ping=True,
                    pool_recycle=3600
                )

            connection_url = URL.create(
                "postgresql+psycopg2",
                username=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database
            )
            return create_engine(
                connection_url,
                poolclass=NullPool,
                pool_pre_ping=True,
                pool_recycle=3600
            )
        else:  # sqlite
            connection_string = f"sqlite:///{self.sqlite_path}"
            return create_engine(
                connection_string,
                connect_args={"check_same_thread": False}
            )

    def connection_label(self) -> str:
        """Return a safe, user-facing connection description."""
        if self.db_type == 'sqlite':
            return f"SQLite file: {self.sqlite_path}"
        return f"{self.db_type}://{self.user}@{self.host}:{self.port}/{self.database}"

    def get_connection(self):
        """Get a raw (non-SQLAlchemy) DBAPI connection for the configured backend"""
        if self.db_type == 'mysql':
            import pymysql
            return pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port,
                charset='utf8mb4'
            )
        elif self.db_type == 'postgresql':
            try:
                import psycopg2
            except ImportError as e:
                raise RuntimeError(
                    "psycopg2-binary is required for PostgreSQL support. "
                    "Install it with: pip install psycopg2-binary"
                ) from e
            return psycopg2.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                dbname=self.database,
                port=self.port
            )
        else:  # sqlite
            import sqlite3
            return sqlite3.connect(self.sqlite_path, check_same_thread=False)

    @staticmethod
    def _validate_query(query: str) -> str:
        """Validate a SQL query before it is allowed to run.

        This is the single choke point that every query - whether typed by
        a user in the Data Explorer / SQL editor, or generated by the LLM
        agent - passes through. Previously this validation didn't exist
        anywhere in the codebase even though Config.FORBIDDEN_SQL was
        defined, so a malicious or LLM-hallucinated query containing e.g.
        DROP/DELETE/UPDATE, or a stacked ';'-separated statement, would be
        executed with no checks at all.

        Raises SQLValidationError if the query is not a safe read-only
        SELECT/WITH statement, or returns the (possibly LIMIT-capped) query
        otherwise.
        """
        if not query or not query.strip():
            raise SQLValidationError("Empty query is not allowed")

        stripped = query.strip()

        # Strip a single trailing semicolon (harmless), then make sure no
        # other statement is stacked behind a ';' (e.g. "SELECT 1; DROP ...").
        body = stripped[:-1] if stripped.endswith(';') else stripped
        if ';' in body:
            raise SQLValidationError(
                "Multiple/stacked SQL statements are not allowed"
            )

        if not re.match(r'^(SELECT|WITH|SHOW|DESCRIBE|DESC|EXPLAIN|PRAGMA)\b', body, re.IGNORECASE):
            raise SQLValidationError("Only read-only SELECT/WITH/SHOW/DESCRIBE/PRAGMA queries are allowed")

        if re.match(r'^PRAGMA\b', body, re.IGNORECASE):
            match = re.match(r'^PRAGMA\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\(|=|$)', body, re.IGNORECASE)
            pragma_name = match.group(1).lower() if match else ""
            allowed_pragmas = {
                "table_info",
                "table_xinfo",
                "foreign_key_list",
                "index_list",
                "index_info",
                "database_list",
                "collation_list",
                "function_list",
                "module_list",
                "pragma_list",
            }
            if pragma_name not in allowed_pragmas:
                raise SQLValidationError(f"PRAGMA {pragma_name or '<unknown>'} is not allowed")

        upper_body = body.upper()
        for keyword in Config.FORBIDDEN_SQL:
            # Match the keyword as a whole word so e.g. "UPDATED_AT" column
            # names don't get falsely flagged because they contain "UPDATE".
            if re.search(rf'\b{re.escape(keyword)}\b', upper_body):
                raise SQLValidationError(f"Forbidden SQL keyword detected: {keyword}")

        # Enforce a sane row cap if the caller didn't already specify one,
        # so a runaway/unbounded query can't pull back the whole table.
        # Only applies to SELECT/WITH - LIMIT isn't valid syntax after
        # SHOW/DESCRIBE/EXPLAIN/PRAGMA.
        if re.match(r'^(SELECT|WITH)\b', body, re.IGNORECASE) and not re.search(r'\bLIMIT\b', upper_body):
            body = f"{body} LIMIT {Config.MAX_QUERY_ROWS}"

        return body

    def execute_query(self, query, params: dict = None):
        """Execute a validated, read-only SQL query and return results as a DataFrame.

        params, if given, is passed through as SQLAlchemy bound parameters
        (e.g. {"name": "Acme"} for a query containing ":name") so caller
        code can safely interpolate *values* without building raw SQL
        strings.
        """
        try:
            safe_query = self._validate_query(query)
            with self.engine.connect() as conn:
                result = conn.execute(text(safe_query), params or {})
                # Get column names
                columns = result.keys()
                # Fetch all rows
                rows = result.fetchall()
                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=columns)
                return df
        except SQLValidationError as e:
            print(f"Query rejected: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()

    def execute_query_strict(self, query, params: dict = None):
        """Execute a validated query and raise errors to the caller.

        The UI-friendly execute_query() intentionally returns an empty
        dataframe on errors. The AI agent needs strict errors so it can ask
        the LLM to repair invalid SQL instead of misreporting "No records".
        """
        safe_query = self._validate_query(query)
        with self.engine.connect() as conn:
            result = conn.execute(text(safe_query), params or {})
            return pd.DataFrame(result.fetchall(), columns=result.keys())

    def execute_raw_query(self, query):
        """Execute raw SQL query via the backend's native DBAPI driver"""
        try:
            safe_query = self._validate_query(query)
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(safe_query)
                    if cursor.description:
                        columns = [desc[0] for desc in cursor.description]
                        rows = cursor.fetchall()
                        return pd.DataFrame(rows, columns=columns)
                    else:
                        conn.commit()
                        return pd.DataFrame()
        except SQLValidationError as e:
            print(f"Query rejected: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()

    def list_tables(self):
        """List all tables in the connected database.

        This was previously called from sql_agent.py (self.db.list_tables())
        but was never defined on this class at all, which meant schema
        auto-detection crashed with an AttributeError as soon as no table
        was cached yet. The query needed to list tables differs per
        backend, which is also why this didn't simply go through
        execute_query.
        """
        try:
            with self.engine.connect() as conn:
                if self.db_type == 'mysql':
                    result = conn.execute(text("SHOW TABLES"))
                elif self.db_type == 'postgresql':
                    result = conn.execute(text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' ORDER BY table_name"
                    ))
                else:  # sqlite
                    result = conn.execute(text(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                        "ORDER BY name"
                    ))
                rows = result.fetchall()
                return pd.DataFrame(rows, columns=["table_name"])
        except Exception as e:
            print(f"Error listing tables: {e}")
            return pd.DataFrame()

    def get_table_schema(self, table_name):
        """Get table schema (column names/types) for the given table"""
        if not re.match(r'^[A-Za-z0-9_]+$', table_name or ''):
            print(f"Error: invalid table name '{table_name}'")
            return pd.DataFrame()

        try:
            with self.engine.connect() as conn:
                if self.db_type == 'mysql':
                    result = conn.execute(text(f"DESCRIBE {table_name}"))
                elif self.db_type == 'postgresql':
                    result = conn.execute(
                        text(
                            "SELECT column_name, data_type, is_nullable, column_default "
                            "FROM information_schema.columns "
                            "WHERE table_name = :table_name "
                            "ORDER BY ordinal_position"
                        ),
                        {"table_name": table_name}
                    )
                else:  # sqlite
                    result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                columns = result.keys()
                rows = result.fetchall()
                return pd.DataFrame(rows, columns=columns)
        except Exception as e:
            print(f"Error getting table schema: {e}")
            return pd.DataFrame()

    def get_sample_data(self, table_name, limit=5):
        """Get sample data from table"""
        if not re.match(r'^[A-Za-z0-9_]+$', table_name or ''):
            print(f"Error: invalid table name '{table_name}'")
            return pd.DataFrame()
        query = f"SELECT * FROM {table_name} LIMIT {int(limit)}"
        return self.execute_query(query)
