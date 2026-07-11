# sql_agent.py
import pandas as pd
import json
import re
from typing import Dict, List, Any
from langchain_core.prompts import ChatPromptTemplate
from llm_manager import LLMManager
from schemas_detector import SchemaDetector
from database_normalisation import DataNormalizer


SQL_GENERATION_TEMPLATE = """You are an expert SQL developer specializing in business intelligence and inventory management.

Database Schema:
{schema_prompt}

User Question:
{question}

Rules:
1. Only generate SQL - no explanations
2. {dialect_rules}
3. Use LIMIT for large result sets (max 1000 rows)
4. Use appropriate JOINs when needed
5. Use aggregate functions (SUM, AVG, COUNT, etc.) for metrics
6. Format numbers with ROUND(..., 2) for currency
7. Use date functions supported by the active database backend
8. Never generate DELETE, UPDATE, DROP, or INSERT statements
9. Use column names exactly as they appear in the schema
10. Add meaningful aliases for calculated columns

Generate only the SQL query:"""

SQL_REPAIR_TEMPLATE = """You generated SQL that failed for this database.

Database dialect: {dialect}
Schema:
{schema_prompt}

User question:
{question}

Failed SQL:
{sql}

Database/validation error:
{error}

Return only one corrected read-only SQL query. Do not use markdown fences."""

INSIGHT_TEMPLATE = """You are a business intelligence analyst. Analyze the following data and provide insights.

Dataset Type: {dataset_type}
SQL Query Used:
{sql}

Query Results (first 30 rows):
{result_table}

Data Summary:
- Total rows returned: {row_count}
- Columns: {columns}

Please provide:
1. Key business insights from this data
2. Notable trends or patterns
3. Actionable recommendations
4. Any anomalies or areas of concern

Format your response in a clear, professional manner with bullet points where appropriate."""

QUERY_SUGGESTION_TEMPLATE = """Given this database schema:
{schema_prompt}

Write a SQL query to answer this question:
Question: {question}

Return ONLY the SQL query, without any additional text."""


class EnhancedSQLAgent:
    def __init__(self, db_connection):
        self.db = db_connection
        self.schema_detector = SchemaDetector()
        self.data_normalizer = DataNormalizer()
        self.dataset_schemas = {}
        self.current_dataset = None
        self.last_error = None

        # LLMManager owns both providers (Gemini primary, Groq fallback) and
        # exposes a single provider-agnostic .invoke(prompt) method.
        self.llm_manager = LLMManager()

        # Prompt templates, built once and reused. We call
        # template.format_messages(...) and pass the result to
        # self.llm_manager.invoke(...) at each call site, rather than piping
        # the template directly into one LLM, so the Gemini/Groq failover
        # inside LLMManager applies uniformly everywhere.
        self.sql_generation_prompt = ChatPromptTemplate.from_template(SQL_GENERATION_TEMPLATE)
        self.sql_repair_prompt = ChatPromptTemplate.from_template(SQL_REPAIR_TEMPLATE)
        self.insight_prompt = ChatPromptTemplate.from_template(INSIGHT_TEMPLATE)
        self.query_suggestion_prompt = ChatPromptTemplate.from_template(QUERY_SUGGESTION_TEMPLATE)

        # Detect schema on initialization
        self._initialize_schema()

    def _initialize_schema(self):
        """Initialize schema detection on startup"""
        try:
            tables = self.db.list_tables()
            if tables is not None and not tables.empty:
                best_table = self._choose_default_table(tables.iloc[:, 0].astype(str).tolist())
                self.detect_schema(best_table)
        except Exception as e:
            print(f"Error initializing schema: {e}")

    def _choose_default_table(self, tables: List[str]) -> str:
        """Pick the most useful user dataset when several tables exist.

        Verification tables and tiny smoke-test tables can sit in the same
        SQLite file during development. Picking the first table alphabetically
        can make the chat agent query the wrong dataset, so prefer wide/high-row
        tables and penalize obvious test table names.
        """
        if not tables:
            return None

        best_table = tables[0]
        best_score = None

        for table in tables:
            row_count = 0
            column_count = 0

            try:
                count_df = self.db.execute_query(f"SELECT COUNT(*) AS row_count FROM {table}")
                if not count_df.empty:
                    row_count = int(count_df.iloc[0, 0])
            except Exception:
                pass

            try:
                schema_df = self.db.get_table_schema(table)
                column_count = len(schema_df) if schema_df is not None else 0
            except Exception:
                pass

            penalty = 0
            lowered = table.lower()
            if lowered.startswith("codex_") or "smoke" in lowered:
                penalty += 1_000_000
            if lowered.endswith("_test") or "_test_" in lowered or lowered.endswith("_ui_path"):
                penalty += 100_000
            if "pipeline_test" in lowered:
                penalty += 100_000

            score = (row_count, column_count, -penalty, table)
            if best_score is None or score > best_score:
                best_score = score
                best_table = table

        return best_table

    def set_current_dataset(self, table_name: str) -> Dict[str, Any]:
        """Set the active table used by chat, KPIs, and analysis."""
        schema = self.detect_schema(table_name)
        if 'error' not in schema:
            self.current_dataset = table_name
        return schema

    def detect_schema(self, table_name: str = None) -> Dict[str, Any]:
        """Detect schema for a specific table or the first available table"""
        if table_name is None:
            if self.current_dataset:
                table_name = self.current_dataset
            else:
                tables = self.db.list_tables()
                if tables.empty:
                    return {'error': 'No tables found in database'}
                table_name = self._choose_default_table(tables.iloc[:, 0].astype(str).tolist())

        if not table_name:
            return {'error': 'No tables found in database'}

        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', table_name):
            return {'error': f'Invalid table name: {table_name}'}

        if table_name in self.dataset_schemas:
            self.current_dataset = table_name
            return self.dataset_schemas[table_name]

        # Get sample data
        sample_query = f"SELECT * FROM {table_name} LIMIT 1000"
        df = self.db.execute_query(sample_query)

        if df.empty:
            return {'error': f'No data found in table {table_name}'}

        # Detect schema
        schema = self.schema_detector.detect_schema(df)
        schema['table_name'] = table_name
        try:
            count_df = self.db.execute_query(f"SELECT COUNT(*) AS row_count FROM {table_name}")
            schema["row_count"] = int(count_df.iloc[0, 0]) if not count_df.empty else len(df)
        except Exception:
            schema["row_count"] = len(df)
        self.dataset_schemas[table_name] = schema
        self.current_dataset = table_name

        return schema

    def _old_detect_schema_removed(self, table_name: str = None) -> Dict[str, Any]:
        """Kept only to make patches explicit; not called."""
        if table_name is None:
            tables = self.db.list_tables()
            if tables.empty:
                return {'error': 'No tables found in database'}
            table_name = tables.iloc[0, 0]

        if table_name in self.dataset_schemas:
            return self.dataset_schemas[table_name]

        # Get sample data
        sample_query = f"SELECT * FROM {table_name} LIMIT 1000"
        df = self.db.execute_query(sample_query)

        if df.empty:
            return {'error': f'No data found in table {table_name}'}

        # Detect schema
        schema = self.schema_detector.detect_schema(df)
        schema['table_name'] = table_name
        schema["row_count"] = len(df)
        self.dataset_schemas[table_name] = schema
        self.current_dataset = table_name

        return schema

    def get_schema_prompt(self, table_name: str = None) -> str:
        """Get a concise schema description for prompts"""
        schema = self.detect_schema(table_name)
        if 'error' in schema:
            return json.dumps({'error': schema['error']})

        # Create a clean schema description
        schema_info = {
            'table_name': schema.get('table_name'),
            'dataset_type': schema.get('dataset_type', 'general'),
            'columns': {
                col: {
                    'type': info.get('data_type'),
                    'sample': info.get('sample_values', [])[:3]
                }
                for col, info in schema.get('columns', {}).items()
            },
            'column_mapping': schema.get('column_mapping', {}),
            'inventory_columns': schema.get('inventory_columns', {}),
            'primary_keys': schema.get('primary_keys', [])
        }

        return json.dumps(schema_info, indent=2)

    def _dialect_rules(self) -> str:
        """Return prompt rules for the active SQL dialect."""
        dialect = getattr(self.db, "db_type", "sqlite")
        if dialect == "mysql":
            return (
                "Use MySQL syntax. Date helpers may include CURDATE(), "
                "DATE_ADD(), DATE_SUB(), DATEDIFF(), and DATE_FORMAT()."
            )
        if dialect == "postgresql":
            return (
                "Use PostgreSQL syntax. Use CURRENT_DATE, INTERVAL values, "
                "DATE_TRUNC(), and TO_CHAR() for date formatting."
            )
        return (
            "Use SQLite syntax. Use date('now'), date(column), strftime(), "
            "and SQLite-compatible LIMIT syntax. Do not use CURDATE(), "
            "DATE_ADD(), DATE_SUB(), DATEDIFF(), or DATE_FORMAT()."
        )

    def generate_sql(self, question: str, schema: Dict[str, Any] = None) -> str:
        """Generate SQL from natural language question"""
        if schema is None:
            schema = self.detect_schema()

        template_sql = self._generate_template_sql(question, schema)
        if template_sql:
            return template_sql

        schema_prompt = self.get_schema_prompt(schema.get('table_name'))

        messages = self.sql_generation_prompt.format_messages(
            schema_prompt=schema_prompt,
            question=question,
            dialect_rules=self._dialect_rules(),
        )

        response = self.llm_manager.invoke(messages)

        print("=" * 60)
        print("SQL Generation Provider:", self.llm_manager.last_provider)
        print("Raw SQL:")
        print(response.content)
        print("=" * 60)

        sql = response.content
        sql = self._clean_generated_sql(response.content)
        sql = response.content



        return sql.strip()

    @staticmethod
    def _clean_generated_sql(sql: str) -> str:
        """Extract runnable SQL from LLM output and markdown fences."""
        sql = (sql or "").strip()

        fenced = re.search(r"```(?:sql|sqlite|mysql|postgresql|postgres)?\s*(.*?)```", sql, re.IGNORECASE | re.DOTALL)
        if fenced:
            sql = fenced.group(1).strip()
        else:
            sql = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", sql).strip()
            sql = re.sub(r"\s*```$", "", sql).strip()

        select_match = re.search(r'\b(SELECT|WITH)\b\s+.*', sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            sql = select_match.group(0).strip()

        return sql.replace("`", "").strip()

    def _generate_template_sql(self, question: str, schema: Dict[str, Any]) -> str:
        """Generate reliable SQL for common BI questions without the LLM."""
        if not schema or 'error' in schema:
            return ""

        q = (question or "").lower()
        table = schema.get('table_name')
        mapping = schema.get('column_mapping', {})
        if not table:
            return ""

        customer_id_col = mapping.get('customer_id')
        customer_name_col = mapping.get('customer_name')
        customer_group_cols = [col for col in [customer_id_col, customer_name_col] if col]
        customer_select = ",\n                    ".join(
            [
                f"{customer_id_col} AS customer_id" if customer_id_col else "",
                f"{customer_name_col} AS customer_name" if customer_name_col else "",
            ]
        ).strip(",\n ")
        customer_group = ", ".join(customer_group_cols)
        sales_col = mapping.get('sales')
        profit_col = mapping.get('profit')
        region_col = mapping.get('region')
        category_col = mapping.get('category')
        product_col = mapping.get('product_name') or mapping.get('product')
        order_id_col = mapping.get('order_id') or mapping.get('transaction_id')
        date_col = mapping.get('order_date') or mapping.get('transaction_date') or mapping.get('date')

        if getattr(self.db, "db_type", "sqlite") == "mysql":
            month_expr = f"DATE_FORMAT({date_col}, '%Y-%m')" if date_col else ""
            year_expr = f"YEAR({date_col})" if date_col else ""
            date_valid_filter = f"{date_col} IS NOT NULL" if date_col else "1=1"
        elif getattr(self.db, "db_type", "sqlite") == "postgresql":
            month_expr = f"TO_CHAR(DATE_TRUNC('month', {date_col}), 'YYYY-MM')" if date_col else ""
            year_expr = f"TO_CHAR(DATE_TRUNC('year', {date_col}), 'YYYY')" if date_col else ""
            date_valid_filter = f"{date_col} IS NOT NULL" if date_col else "1=1"
        else:
            # SQLite's strftime() only handles ISO dates (YYYY-MM-DD).
            # The Superstore dataset (and many real CSVs) store dates as
            # MM/DD/YYYY. We use a CASE expression that detects the format
            # by pattern and applies the right extraction, so both ISO and
            # MM/DD/YYYY dates are handled correctly in the same table.
            _iso = f"{date_col} LIKE '____-__-__%'"
            year_expr = (
                f"CASE WHEN {_iso} "
                f"THEN strftime('%Y', {date_col}) "
                f"ELSE substr({date_col}, -4) END"
            ) if date_col else ""
            month_expr = (
                f"CASE WHEN {_iso} "
                f"THEN strftime('%Y-%m', {date_col}) "
                f"ELSE substr({date_col}, -4) || '-' || "
                f"printf('%02d', CAST(substr({date_col}, 1, instr({date_col}, '/') - 1) AS INTEGER)) END"
            ) if date_col else ""
            date_valid_filter = f"{date_col} IS NOT NULL" if date_col else "1=1"

        year_match = re.search(r'\b(20\d{2}|19\d{2})\b', q)
        requested_year = year_match.group(1) if year_match else None
        if requested_year and date_col and sales_col and any(term in q for term in ("total revenue", "revenue generated", "sales generated", "total sales")):
            profit_select = f",\n                    ROUND(SUM({profit_col}), 2) AS total_profit" if profit_col and "profit" in q else ""
            # LIMIT 1 is explicit here (not left for the query validator to
            # add) because this query is filtered to a single year and
            # grouped by that same year, so it can only ever return one row.
            # If we leave it unlimited, the validator appends "LIMIT 1000"
            # to the executed SQL, and the insight LLM then sees that
            # clause and wrongly narrates the result as "a partial figure
            # from only 1000 rows" even though SUM() already ran over the
            # whole table before any row-limit applies.
            return f"""
                SELECT
                    {year_expr} AS year,
                    ROUND(SUM({sales_col}), 2) AS total_revenue
                    {profit_select}
                FROM {table}
                WHERE {year_expr} = '{requested_year}'
                GROUP BY {year_expr}
                LIMIT 1
            """

        if requested_year and date_col and sales_col and profit_col and any(term in q for term in ("gross margin", "margin percentage", "profit margin")):
            return f"""
                SELECT
                    {year_expr} AS year,
                    ROUND(SUM({sales_col}), 2) AS total_revenue,
                    ROUND(SUM({profit_col}), 2) AS total_profit,
                    ROUND(SUM({profit_col}) * 100.0 / NULLIF(SUM({sales_col}), 0), 2) AS gross_margin_percentage
                FROM {table}
                WHERE {year_expr} = '{requested_year}'
                GROUP BY {year_expr}
                LIMIT 1
            """

        if requested_year and date_col and profit_col and "profit" in q:
            return f"""
                SELECT
                    {year_expr} AS year,
                    ROUND(SUM({profit_col}), 2) AS total_profit
                FROM {table}
                WHERE {year_expr} = '{requested_year}'
                GROUP BY {year_expr}
                LIMIT 1
            """

        # Bare aggregate totals — the sidebar "KPI Shortcuts" send these exact
        # labels ("Total Revenue", "Total Profit", "Total Orders", "Total
        # Customers") with no year and no other qualifier, so the year-scoped
        # rules above never fire for them. "total" is a safe, unambiguous
        # keyword here: none of the other templates' question text contains
        # it, so these can't misfire on "Monthly Sales Trend", "YoY Revenue",
        # etc.
        #
        # LIMIT 1 is explicit on all four of these for the same reason as
        # above: they have no GROUP BY at all, so they always return exactly
        # one row. Without an explicit LIMIT, the query validator appends its
        # own "LIMIT 1000", and the insight LLM then (incorrectly) treats
        # that as evidence the aggregate only covers the first 1000 rows.
        if "total" in q and "revenue" in q and sales_col:
            return f"""
                SELECT ROUND(SUM({sales_col}), 2) AS total_revenue
                FROM {table}
                LIMIT 1
            """

        if "total" in q and "profit" in q and profit_col:
            return f"""
                SELECT ROUND(SUM({profit_col}), 2) AS total_profit
                FROM {table}
                LIMIT 1
            """

        if "total" in q and "order" in q and order_id_col:
            return f"""
                SELECT COUNT(DISTINCT {order_id_col}) AS total_orders
                FROM {table}
                LIMIT 1
            """

        if "total" in q and "customer" in q and (customer_id_col or customer_group):
            count_col = customer_id_col or customer_group_cols[0]
            return f"""
                SELECT COUNT(DISTINCT {count_col}) AS total_customers
                FROM {table}
                LIMIT 1
            """

        # "Top Customers by Revenue" ranks by revenue (default); "Top Customers
        # by Profit" ranks by profit instead. Only treated as profit-primary
        # when "profit" is present and "revenue"/"sales" is not, so plain
        # "top 5 customers by revenue" keeps its original behavior.
        if "top" in q and "customer" in q and customer_group and (sales_col or profit_col):
            limit_match = re.search(r'\btop\s+(\d+)\b', q)
            limit = int(limit_match.group(1)) if limit_match else 5
            limit = max(1, min(limit, 100))
            rank_by_profit = "profit" in q and not any(word in q for word in ("revenue", "sales")) and profit_col
            if rank_by_profit:
                primary_col, primary_name = profit_col, "total_profit"
                secondary_select = f",\n                    ROUND(SUM({sales_col}), 2) AS total_revenue" if sales_col else ""
            else:
                primary_col, primary_name = sales_col, "total_revenue"
                secondary_select = f",\n                    ROUND(SUM({profit_col}), 2) AS total_profit" if profit_col else ""
            return f"""
                SELECT
                    {customer_select},
                    ROUND(SUM({primary_col}), 2) AS {primary_name}
                    {secondary_select}
                FROM {table}
                GROUP BY {customer_group}
                ORDER BY {primary_name} DESC
                LIMIT {limit}

            """

        # "Top Products" — deliberately does NOT require a "revenue"/"sales"
        # keyword, since the Quick Analysis button sends the bare label
        # "Top Products" with no metric word in it. Revenue is the sensible
        # default ranking metric when none is specified.
        if "top" in q and "product" in q and product_col and sales_col:
            limit_match = re.search(r'\btop\s+(\d+)\b', q)
            limit = int(limit_match.group(1)) if limit_match else 5
            limit = max(1, min(limit, 100))
            profit_select = f",\n                    ROUND(SUM({profit_col}), 2) AS total_profit" if profit_col else ""
            return f"""
                SELECT
                    {product_col} AS product,
                    ROUND(SUM({sales_col}), 2) AS total_revenue
                    {profit_select}
                FROM {table}
                GROUP BY {product_col}
                ORDER BY total_revenue DESC
                LIMIT {limit}
            """

        # True month-over-month request: compute an actual delta with LAG(),
        # same treatment as the YoY block below. Checked BEFORE the generic
        # monthly-trend rule so "MoM Revenue" doesn't silently fall through to
        # a plain monthly total with no month-over-month comparison at all.
        if any(term in q for term in ("month over month", "mom")) and date_col and (sales_col or profit_col):
            metric_col = profit_col if ("profit" in q and profit_col) else sales_col
            metric_name = "profit" if metric_col is profit_col else "revenue"
            return f"""
                WITH monthly AS (
                    SELECT
                        {month_expr} AS month,
                        ROUND(SUM({metric_col}), 2) AS {metric_name}
                    FROM {table}
                    WHERE {date_valid_filter}
                    GROUP BY {month_expr}
                )
                SELECT
                    month,
                    {metric_name},
                    ROUND({metric_name} - LAG({metric_name}) OVER (ORDER BY month), 2) AS mom_{metric_name}_change,
                    ROUND(
                        ({metric_name} - LAG({metric_name}) OVER (ORDER BY month))
                        / NULLIF(LAG({metric_name}) OVER (ORDER BY month), 0) * 100,
                        2
                    ) AS mom_{metric_name}_change_pct
                FROM monthly
                ORDER BY month
            """

        # Plain monthly trend (no explicit MoM comparison requested). Now
        # recognizes "profit" as its own metric instead of only "revenue"/
        # "sales" — this is what makes "Monthly Profit Trend" work, since
        # that button's label has no "revenue"/"sales" word in it at all.
        if "monthly" in q and date_col and (sales_col or profit_col):
            want_sales = any(term in q for term in ("revenue", "sales"))
            want_profit = "profit" in q
            if not want_sales and not want_profit:
                want_sales = True  # default metric when none is named
            select_parts = [f"{month_expr} AS month"]
            if want_sales and sales_col:
                select_parts.append(f"ROUND(SUM({sales_col}), 2) AS revenue")
            if want_profit and profit_col:
                select_parts.append(f"ROUND(SUM({profit_col}), 2) AS profit")
            select_clause = ",\n                    ".join(select_parts)
            return f"""
                SELECT
                    {select_clause}
                FROM {table}
                WHERE {date_valid_filter}
                GROUP BY {month_expr}
                ORDER BY month
            """

        if any(term in q for term in ("year over year", "yoy")) and date_col and (sales_col or profit_col):
            metric_col = profit_col if ("profit" in q and profit_col) else sales_col
            metric_name = "profit" if metric_col is profit_col else "revenue"
            return f"""
                WITH yearly AS (
                    SELECT
                        {year_expr} AS year,
                        ROUND(SUM({metric_col}), 2) AS {metric_name}
                    FROM {table}
                    WHERE {date_valid_filter}
                    GROUP BY {year_expr}
                )
                SELECT
                    year,
                    {metric_name},
                    ROUND({metric_name} - LAG({metric_name}) OVER (ORDER BY year), 2) AS yoy_{metric_name}_change,
                    ROUND(
                        ({metric_name} - LAG({metric_name}) OVER (ORDER BY year))
                        / NULLIF(LAG({metric_name}) OVER (ORDER BY year), 0) * 100,
                        2
                    ) AS yoy_{metric_name}_change_pct
                FROM yearly
                ORDER BY year
            """

        if "sales" in q and "region" in q and region_col and sales_col:
            return f"""
                SELECT
                    {region_col} AS region,
                    ROUND(SUM({sales_col}), 2) AS total_sales
                FROM {table}
                GROUP BY {region_col}
                ORDER BY total_sales DESC
                LIMIT 100
            """

        # Category breakdown — now recognizes "sales" as its own metric
        # instead of only "profit". This is what makes "Sales by Category"
        # work, since the old rule required the word "profit" to be present.
        if "category" in q and category_col and (sales_col or profit_col):
            want_sales = any(term in q for term in ("sales", "revenue"))
            want_profit = "profit" in q
            if not want_sales and not want_profit:
                want_sales = True  # default metric when none is named
            select_parts = [f"{category_col} AS category"]
            order_col = None
            if want_sales and sales_col:
                select_parts.append(f"ROUND(SUM({sales_col}), 2) AS total_sales")
                order_col = "total_sales"
            if want_profit and profit_col:
                select_parts.append(f"ROUND(SUM({profit_col}), 2) AS total_profit")
                order_col = order_col or "total_profit"
            select_clause = ",\n                    ".join(select_parts)
            return f"""
                SELECT
                    {select_clause}
                FROM {table}
                GROUP BY {category_col}
                ORDER BY {order_col} DESC
                LIMIT 100
            """

        # General "Customer Analysis" — a broader per-customer summary for
        # when the question doesn't ask for a ranked "top N" list (that case
        # is handled by the more specific rule above). This is what makes the
        # bare "Customer Analysis" button label work instead of falling
        # through to the LLM.
        if "customer" in q and "top" not in q and customer_group and sales_col:
            profit_select = f",\n                    ROUND(SUM({profit_col}), 2) AS total_profit" if profit_col else ""
            orders_select = f",\n                    COUNT(DISTINCT {order_id_col}) AS order_count" if order_id_col else ""
            return f"""
                SELECT
                    {customer_select},
                    ROUND(SUM({sales_col}), 2) AS total_revenue
                    {profit_select}
                    {orders_select}
                FROM {table}
                GROUP BY {customer_group}
                ORDER BY total_revenue DESC
                LIMIT 20
            """

        return ""

    def _generate_sql_with_repair(self, question: str, schema: Dict[str, Any], max_attempts: int = 2) -> str:
        """Generate SQL and retry once with execution feedback if needed."""
        sql = self.generate_sql(question, schema)

        for attempt in range(max_attempts):
            try:
                safe_sql = self.db._validate_query(sql)
                result = self.db.execute_query_strict(sql)
                return safe_sql, result
            except Exception as e:
                if attempt >= max_attempts - 1:
                    raise

                messages = self.sql_repair_prompt.format_messages(
                    dialect=getattr(self.db, "db_type", "sqlite"),
                    schema_prompt=self.get_schema_prompt(schema.get('table_name')),
                    question=question,
                    sql=sql,
                    error=str(e),
                )
                response = self.llm_manager.invoke(messages)
                sql = self._clean_generated_sql(response.content)

        return sql, pd.DataFrame()

    def execute_question(self, question: str) -> tuple:
        """Execute a natural language question against the database"""
        schema = self.detect_schema()

        if 'error' in schema:
            return pd.DataFrame(), f"-- Error: {schema['error']}"

        try:
            sql, result = self._generate_sql_with_repair(question, schema)
            return result, sql
        except Exception as e:
            return pd.DataFrame(), f"-- Error generating SQL: {str(e)}"

    def answer_question(self, question: str) -> str:
        """Answer a natural language question with business insights"""
        df, sql = self.execute_question(question)
        print("=" * 60)
        print("SQL Returned:")
        print(sql)
        print("Rows Returned:", len(df))
        print("=" * 60)

        if df.empty:
            return f"No records found.\n\nSQL attempted:\n```sql\n{sql}\n```"

        # Get schema for context
        schema = self.detect_schema()
        dataset_type = schema.get('dataset_type', 'general')

        result_table = df.head(30).to_markdown(index=False)

        try:
            messages = self.insight_prompt.format_messages(
                dataset_type=dataset_type,
                sql=sql,
                result_table=df.head(30).to_markdown(),
                row_count=len(df),
                columns=', '.join(df.columns.tolist()),
            )
            print("Calling LLM for insights...")

            response = self.llm_manager.invoke(messages)

            print("Insight Provider:", self.llm_manager.last_provider)
            print("Insight:")
            print(response.content)

            provider = self.llm_manager.last_provider or "llm"
            return (
                f"SQL used:\n```sql\n{sql}\n```\n\n"
                f"Result:\n{result_table}\n\n"
                f"{provider.title()} insight:\n{response.content}"
            )
        except Exception as e:
            return (
                f"Query ran successfully, but insight generation failed: {e}\n\n"
                f"SQL used:\n```sql\n{sql}\n```\n\n"
                f"Result:\n{result_table}"
            )

    def analyze_dataset(self, table_name: str = None) -> str:
        """Analyze dataset and return insights"""
        schema = self.detect_schema(table_name)

        if 'error' in schema:
            return schema['error']

        dataset_type = schema.get('dataset_type', 'general')

        analysis = f"""
📊 Dataset Analysis: {schema['table_name']}

📋 Table Overview:
- Total Columns: {len(schema['columns'])}
- Total Rows: {schema.get('row_count', 'Unknown')}
- Dataset Type: {dataset_type.replace('_', ' ').title()}

🔍 Column Types:
- Date Columns: {', '.join(schema.get('date_columns', [])[:5])}
- Numeric Columns: {', '.join(schema.get('numeric_columns', [])[:5])}
- String Columns: {', '.join(schema.get('string_columns', [])[:5])}

🏷️ Column Mapping:
{self._format_column_mapping(schema.get('column_mapping', {}))}

📈 Suggested Metrics:
{self._format_suggested_metrics(schema.get('suggested_metrics', []))}

🔑 Primary Keys:
{', '.join(schema.get('primary_keys', ['None identified']))}

Missing Values:
{self._format_missing_values(schema.get('null_counts', {}))}
"""

        # Add inventory-specific analysis
        if dataset_type == 'inventory_management':
            analysis += f"""

📦 Inventory Analysis:
{self._format_inventory_analysis(schema)}
"""

        return analysis

    def _format_column_mapping(self, mapping: Dict[str, str]) -> str:
        """Format column mapping for display"""
        if not mapping:
            return "No mappings identified"

        lines = []
        for std_name, actual_col in mapping.items():
            lines.append(f"  - {std_name}: {actual_col}")
        return '\n'.join(lines)

    def _format_suggested_metrics(self, metrics: List[Dict]) -> str:
        """Format suggested metrics for display"""
        if not metrics:
            return "No metrics suggested"

        lines = []
        for metric in metrics[:10]:
            lines.append(f"  - {metric.get('name')}: {metric.get('description')}")
        return '\n'.join(lines)

    def _format_missing_values(self, null_counts: Dict[str, int]) -> str:
        """Format missing values for display"""
        if not null_counts:
            return "No missing values detected"

        missing = {k: v for k, v in null_counts.items() if v > 0}
        if not missing:
            return "No missing values detected"

        lines = []
        for col, count in list(missing.items())[:5]:
            lines.append(f"  - {col}: {count} null values")
        return '\n'.join(lines)

    def _format_inventory_analysis(self, schema: Dict[str, Any]) -> str:
        """Format inventory-specific analysis"""
        inventory_cols = schema.get('inventory_columns', {})
        inventory_metrics = schema.get('inventory_metrics', {})

        lines = []
        lines.append("🔹 Inventory Columns Detected:")

        for col_type, col_name in inventory_cols.items():
            if col_name:
                lines.append(f"  - {col_type.replace('_', ' ').title()}: {col_name}")

        if inventory_metrics:
            lines.append("\n🔹 Inventory Metrics:")
            for metric, value in inventory_metrics.items():
                if isinstance(value, (int, float)):
                    if 'value' in metric.lower():
                        lines.append(f"  - {metric.replace('_', ' ').title()}: ${value:,.2f}")
                    else:
                        lines.append(f"  - {metric.replace('_', ' ').title()}: {value:,}")
                else:
                    lines.append(f"  - {metric.replace('_', ' ').title()}: {value}")

        return '\n'.join(lines)

    def get_query_suggestions(self, question: str) -> str:
        """Get SQL query suggestions based on question"""
        try:
            schema = self.detect_schema()
            schema_prompt = self.get_schema_prompt(schema.get('table_name'))

            messages = self.query_suggestion_prompt.format_messages(
                schema_prompt=schema_prompt,
                question=question,
            )
            response = self.llm_manager.invoke(messages)
            sql = response.content
            sql = sql.replace("```sql", "").replace("```", "").strip()
            return sql
        except Exception as e:
            return f"Could not generate SQL query: {str(e)}"

    def get_predefined_questions(self) -> List[str]:
        """Get list of predefined NLP questions based on dataset type"""
        schema = self.detect_schema()
        dataset_type = schema.get('dataset_type', 'general')

        base_questions = [
            "What are the top 5 customers by revenue?",
            "Show me total sales by region",
            "What is the total profit for each category?",
            "Which products have the highest profit margin?",
            "Show me monthly sales trend for the last year",
            "What is the average order value by customer segment?",
            "Which sub-categories are generating the most revenue?",
            "Show me the top 10 products by sales",
            "What are the year-over-year sales growth trends?",
            "Which customers have the highest lifetime value?",
            "What is the profit margin by region?",
            "Show me the distribution of orders by shipping mode",
            "Which products have negative profit margin?",
            "What is the customer retention rate?",
            "Show me sales performance by quarter"
        ]

        # Inventory-specific questions
        inventory_questions = [
            "What is the current stock status?",
            "Which items are low in stock?",
            "What is the total inventory value?",
            "Show me items that need reordering",
            "What are the slow moving items?",
            "Which products have the highest turnover?",
            "Show me stock distribution by warehouse",
            "What is the supplier performance?",
            "Which items are expiring soon?",
            "Show me ABC classification of inventory",
            "What is the average stock turnover rate?",
            "Which products are overstocked?",
            "Show me inventory value by category",
            "What is the reorder point analysis?",
            "Show me stock levels by product category"
        ]

        if dataset_type == 'inventory_management':
            return inventory_questions + base_questions

        return base_questions

    def get_kpi_list(self) -> List[str]:
        """Get list of available KPIs based on current dataset"""
        schema = self.detect_schema()
        if 'error' in schema:
            return []

        metrics = schema.get('suggested_metrics', [])
        return [metric.get('name') for metric in metrics if metric.get('name')]

    def _get_available_tables(self) -> List[str]:
        """Get a plain list of table names available in the database.

        This is used by the Data Explorer page in app.py
        (st.session_state.agent._get_available_tables()), which previously
        called a method that didn't exist anywhere in this class.
        """
        try:
            tables_df = self.db.list_tables()
            if tables_df is None or tables_df.empty:
                return []
            return tables_df.iloc[:, 0].tolist()
        except Exception as e:
            print(f"Error getting available tables: {e}")
            return []

    def load_dataset(self, df: pd.DataFrame, table_name: str, already_normalized: bool = False) -> bool:
        """Load dataset into SQL database."""

        try:
            self.last_error = None
            if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', table_name or ''):
                raise ValueError("Table name must start with a letter/underscore and contain only letters, numbers, and underscores.")

            # Detect schema
            schema = self.schema_detector.detect_schema(df)

            # Normalize dataframe unless the UI already ran the selected
            # cleaning pipeline and handed us the cleaned dataframe.
            if already_normalized:
                df_normalized = df.copy()
            else:
                df_normalized = self.data_normalizer.normalize_dataframe(df, schema)

            # Write inside a transaction. SQLite has a fairly low bound
            # parameter limit, so a large CSV cannot be inserted as one
            # giant multi-value INSERT. Chunking keeps the same pipeline
            # working for SQLite, MySQL, and PostgreSQL.
            column_count = max(len(df_normalized.columns), 1)
            if getattr(self.db, "db_type", "sqlite") == "sqlite":
                insert_method = None
                chunksize = max(1, min(500, 900 // column_count))
            else:
                insert_method = "multi"
                chunksize = max(1, min(1000, 5000 // column_count))

            with self.db.engine.begin() as conn:
                df_normalized.to_sql(
                    name=table_name,
                    con=conn,
                    if_exists="replace",
                    index=False,
                    method=insert_method,
                    chunksize=chunksize
                )

            # Verify table creation
            tables = self.db.list_tables()

            if tables.empty:
                raise RuntimeError("No tables found after loading dataset.")

            if table_name not in tables.iloc[:, 0].astype(str).tolist():
                raise RuntimeError(f"Table '{table_name}' was not created.")

            # Refresh schema
            schema = self.detect_schema(table_name)

            self.dataset_schemas[table_name] = schema
            self.current_dataset = table_name

            return True

        except Exception:
            import traceback

            self.last_error = traceback.format_exc()
            print("=" * 80)
            print(f"FAILED TO LOAD DATASET: {table_name}")
            print(self.last_error)
            print("=" * 80)

            return False



    def execute_kpi_query(self, query_name: str) -> str:
        """Execute predefined KPI queries based on current dataset"""
        result = self.execute_kpi_dataframe(query_name)
        if isinstance(result, str):
            return result
        if not result.empty:
            return result.to_string(index=False)
        return "No data found"

    def execute_kpi_dataframe(self, query_name: str):
        """Execute predefined KPI query and return a DataFrame when possible."""
        schema = self.detect_schema()

        if 'error' in schema:
            return schema['error']

        metrics = schema.get('suggested_metrics', [])

        for metric in metrics:
            if metric.get('name') == query_name:
                query = metric.get('query', '').replace('{table}', schema.get('table_name'))
                return self.db.execute_query(query)

        available = '\n'.join([f"  - {m.get('name')}" for m in metrics])
        return f"Unknown KPI: {query_name}\nAvailable KPIs:\n{available}"

    def inventory_analysis(self, analysis_type: str = 'stock_status') -> pd.DataFrame:
        """Perform inventory-specific analysis.

        Returns a DataFrame with the analysis results, or a single-column
        DataFrame with a 'message' column describing why no results are
        available (no inventory columns detected, unknown analysis type,
        no matching data, etc.) so callers always get a consistent type
        back rather than sometimes a string and sometimes a DataFrame.
        """
        schema = self.detect_schema()

        if 'error' in schema:
            return pd.DataFrame({'message': [schema['error']]})

        inventory_cols = schema.get('inventory_columns', {})
        table = schema.get('table_name')

        if not inventory_cols or not table:
            return pd.DataFrame({'message': ['No inventory columns detected in the dataset']})

        # Build inventory queries based on available columns
        analysis_queries = {}

        # Stock status
        stock_col = inventory_cols.get('stock_level')
        reorder_col = inventory_cols.get('reorder_point')

        if stock_col:
            if reorder_col:
                analysis_queries['stock_status'] = f"""
                    SELECT 
                        CASE 
                            WHEN {stock_col} = 0 THEN 'Out of Stock'
                            WHEN {stock_col} <= {reorder_col} THEN 'Low Stock'
                            WHEN {stock_col} > {reorder_col} * 3 THEN 'Overstock'
                            ELSE 'In Stock'
                        END as Stock_Status,
                        COUNT(*) as Product_Count,
                        ROUND(AVG({stock_col}), 2) as Avg_Stock
                    FROM {table}
                    GROUP BY Stock_Status
                """
            else:
                analysis_queries['stock_status'] = f"""
                    SELECT 
                        CASE 
                            WHEN {stock_col} = 0 THEN 'Out of Stock'
                            WHEN {stock_col} <= 10 THEN 'Low Stock'
                            WHEN {stock_col} > 100 THEN 'Overstock'
                            ELSE 'In Stock'
                        END as Stock_Status,
                        COUNT(*) as Product_Count,
                        ROUND(AVG({stock_col}), 2) as Avg_Stock
                    FROM {table}
                    GROUP BY Stock_Status
                """

        # Reorder point analysis
        if reorder_col and stock_col:
            analysis_queries['reorder_analysis'] = f"""
                SELECT 
                    CASE 
                        WHEN {stock_col} <= {reorder_col} THEN 'Needs Reorder'
                        ELSE 'Sufficient Stock'
                    END as Reorder_Status,
                    COUNT(*) as Product_Count,
                    ROUND(AVG({stock_col}), 2) as Avg_Stock,
                    ROUND(AVG({reorder_col}), 2) as Avg_Reorder_Point
                FROM {table}
                GROUP BY Reorder_Status
            """

        # Inventory value
        if stock_col:
            # BUG FIX: this used to do `for col in inventory_cols: if col in
            # ['cost','price','unit_cost']`, but inventory_cols' keys are
            # always the fixed set ('stock_level', 'reorder_point', etc.)
            # from schemas_detector.py's _detect_inventory_columns() -- they
            # are never literally 'cost'/'price'/'unit_cost', so that loop
            # could never match anything and cost_col was always None. This
            # scans the real numeric column names instead, matching the same
            # 'cost'/'price' substring approach schemas_detector.py itself
            # uses in _calculate_inventory_metrics().
            cost_col = None
            for col in schema.get('numeric_columns', []):
                if col != stock_col and ('cost' in col.lower() or 'price' in col.lower()):
                    cost_col = col
                    break

            # BUG FIX: this analysis is labeled "Inventory Value by Category"
            # in the UI, but it never actually grouped by a category column --
            # it always returned one single table-wide total with the raw
            # TABLE NAME hardcoded into a column literally called 'Category'
            # (e.g. "u1_dummy_inventory_8000" shown as if it were a product
            # category). Now it groups by the dataset's real category column
            # when the schema detector found one, and only falls back to a
            # single-row grand total (honestly labeled) when it didn't.
            category_col = schema.get('column_mapping', {}).get('category')
            value_expr = f"{stock_col} * {cost_col}" if cost_col else stock_col
            value_name = "Value" if cost_col else "Quantity"

            if category_col:
                analysis_queries['inventory_value'] = f"""
                    SELECT 
                        {category_col} as Category,
                        ROUND(SUM({value_expr}), 2) as Total_{value_name},
                        ROUND(AVG({value_expr}), 2) as Avg_Item_{value_name},
                        COUNT(*) as Total_Items
                    FROM {table}
                    GROUP BY {category_col}
                    ORDER BY Total_{value_name} DESC
                """
            else:
                analysis_queries['inventory_value'] = f"""
                    SELECT 
                        'All Products' as Scope,
                        ROUND(SUM({value_expr}), 2) as Total_{value_name},
                        ROUND(AVG({value_expr}), 2) as Avg_Item_{value_name},
                        COUNT(*) as Total_Items
                    FROM {table}
                """

        # Turnover analysis
        turnover_col = inventory_cols.get('turnover')
        if turnover_col:
            analysis_queries['turnover_analysis'] = f"""
                SELECT 
                    CASE 
                        WHEN {turnover_col} < 1 THEN 'Slow Moving'
                        WHEN {turnover_col} BETWEEN 1 AND 5 THEN 'Medium Moving'
                        WHEN {turnover_col} > 5 THEN 'Fast Moving'
                    END as Turnover_Category,
                    COUNT(*) as Product_Count,
                    ROUND(AVG({turnover_col}), 2) as Avg_Turnover
                FROM {table}
                GROUP BY Turnover_Category
            """

        # Supplier analysis
        supplier_col = inventory_cols.get('supplier')
        if supplier_col and stock_col:
            analysis_queries['supplier_analysis'] = f"""
                SELECT 
                    {supplier_col} as Supplier,
                    COUNT(*) as Product_Count,
                    ROUND(SUM({stock_col}), 2) as Total_Stock
                FROM {table}
                GROUP BY {supplier_col}
                ORDER BY Total_Stock DESC
                LIMIT 10
            """

        # Warehouse analysis
        warehouse_col = inventory_cols.get('warehouse')
        if warehouse_col and stock_col:
            analysis_queries['warehouse_analysis'] = f"""
                SELECT 
                    {warehouse_col} as Warehouse,
                    COUNT(*) as Product_Count,
                    ROUND(SUM({stock_col}), 2) as Total_Stock
                FROM {table}
                GROUP BY {warehouse_col}
                ORDER BY Total_Stock DESC
            """

        # Expiry analysis
        expiry_col = inventory_cols.get('expiry_date')
        if expiry_col:
            if getattr(self.db, "db_type", "sqlite") == "mysql":
                expiry_case = (
                    f"WHEN {expiry_col} < CURDATE() THEN 'Expired' "
                    f"WHEN {expiry_col} BETWEEN CURDATE() AND DATE_ADD(CURDATE(), INTERVAL 30 DAY) THEN 'Expiring Soon'"
                )
            elif getattr(self.db, "db_type", "sqlite") == "postgresql":
                expiry_case = (
                    f"WHEN {expiry_col} < CURRENT_DATE THEN 'Expired' "
                    f"WHEN {expiry_col} BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '30 days' THEN 'Expiring Soon'"
                )
            else:
                expiry_case = (
                    f"WHEN date({expiry_col}) < date('now') THEN 'Expired' "
                    f"WHEN date({expiry_col}) BETWEEN date('now') AND date('now', '+30 days') THEN 'Expiring Soon'"
                )

            analysis_queries['expiry_analysis'] = f"""
                SELECT 
                    CASE 
                        {expiry_case}
                        ELSE 'Valid'
                    END as Expiry_Status,
                    COUNT(*) as Product_Count
                FROM {table}
                WHERE {expiry_col} IS NOT NULL
                GROUP BY Expiry_Status
            """

        if analysis_type not in analysis_queries:
            return pd.DataFrame({
                'message': [f"Unknown analysis type: {analysis_type}. "
                            f"Available types: {', '.join(analysis_queries.keys())}"]
            })

        result = self.db.execute_query(analysis_queries[analysis_type])
        if not result.empty:
            return result
        return pd.DataFrame({'message': ['No data found']})

    def abc_analysis(self, criteria: str = 'value') -> pd.DataFrame:
        """Perform ABC classification analysis.

        Returns a DataFrame (or a single-column 'message' DataFrame
        explaining why no results are available), for the same reason
        described in inventory_analysis().
        """
        schema = self.detect_schema()

        if 'error' in schema:
            return pd.DataFrame({'message': [schema['error']]})

        inventory_cols = schema.get('inventory_columns', {})
        table = schema.get('table_name')
        mapping = schema.get('column_mapping', {})

        if not table or not inventory_cols:
            return pd.DataFrame({'message': ['No inventory columns detected']})

        product_col = inventory_cols.get('product_name') or mapping.get('product_name') or 'Product_Name'
        stock_col = inventory_cols.get('stock_level')

        if not stock_col:
            return pd.DataFrame({'message': ['No stock level column found']})

        # Determine criteria column
        if criteria == 'value':
            # Same bug fix as inventory_analysis() above: scan actual numeric
            # column names for a 'cost'/'price' substring instead of checking
            # whether inventory_cols' fixed dict keys equal those words
            # (which they never do).
            cost_col = None
            for col in schema.get('numeric_columns', []):
                if col != stock_col and ('cost' in col.lower() or 'price' in col.lower()):
                    cost_col = col
                    break

            if cost_col:
                value_col = f"{stock_col} * {cost_col}"
                value_name = "Inventory_Value"
            else:
                value_col = stock_col
                value_name = "Quantity"
        else:
            value_col = stock_col
            value_name = "Quantity"

        # BUG FIX / feature gap: this used to end with a GROUP BY ABC_Class
        # that only returned 3 rows (one per class) with counts and totals --
        # there was no way to see WHICH products actually fell into A, B, or
        # C. Now it returns the real per-product breakdown instead, capped to
        # the top 50 products *within each class* (via ROW_NUMBER, not a
        # single overall LIMIT) so Class C isn't silently starved out by a
        # flat cutoff sorted purely by value -- every class gets a fair,
        # representative sample of its own top contributors.
        query = f"""
            WITH product_metrics AS (
                SELECT 
                    {product_col} as Product,
                    ROUND(SUM({value_col}), 2) as Total_{value_name},
                    ROUND(SUM({value_col}) / SUM(SUM({value_col})) OVER (), 4) as Percentage_Contribution,
                    ROUND(SUM(SUM({value_col})) OVER (ORDER BY SUM({value_col}) DESC) / SUM(SUM({value_col})) OVER (), 4) as Cumulative_Percentage
                FROM {table}
                GROUP BY {product_col}
            ),
            abc_classification AS (
                SELECT 
                    Product,
                    Total_{value_name},
                    Percentage_Contribution,
                    Cumulative_Percentage,
                    CASE 
                        WHEN Cumulative_Percentage <= 0.8 THEN 'A'
                        WHEN Cumulative_Percentage <= 0.95 THEN 'B'
                        ELSE 'C'
                    END as ABC_Class
                FROM product_metrics
            ),
            ranked AS (
                SELECT
                    ABC_Class,
                    Product,
                    Total_{value_name},
                    ROUND(Percentage_Contribution * 100, 2) as Pct_Contribution,
                    ROW_NUMBER() OVER (PARTITION BY ABC_Class ORDER BY Total_{value_name} DESC) as rn
                FROM abc_classification
            )
            SELECT
                ABC_Class,
                Product,
                Total_{value_name},
                Pct_Contribution
            FROM ranked
            WHERE rn <= 50
            ORDER BY ABC_Class, Total_{value_name} DESC
        """

        result = self.db.execute_query(query)
        if not result.empty:
            return result
        return pd.DataFrame({'message': ['No data found for ABC analysis']})

    def query_database(self, query: str) -> str:
        """Execute custom SQL query"""
        # Validate query is SELECT
        if not query.strip().upper().startswith('SELECT') and not query.strip().upper().startswith('WITH'):
            return "Only SELECT queries are allowed"

        result = self.db.execute_query(query)
        if not result.empty:
            return result.to_string(index=False)
        return "No results found"
