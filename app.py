# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from database_connection import DatabaseConnection
from sql_agent import EnhancedSQLAgent
from schemas_detector import SchemaDetector
from database_normalisation import DataNormalizer
from config import Config
import json
import re
from typing import Dict

from pydantic import ValidationError
from auth import (
    init_auth_db, create_user, authenticate_user,
    create_access_token, decode_token,
    scoped_table_name, strip_user_prefix, filter_user_tables,
)
from models import UserCreate, UserLogin

# Page configuration
st.set_page_config(
    page_title="Universal SQL Database Agent - Finance & Inventory",
    page_icon="📊",
    layout="wide"
)

SUPPORTED_DB_BACKENDS = ["sqlite", "mysql", "postgresql"]

# Boot the user-accounts database (creates table if first run)
init_auth_db()


def _current_user():
    """Return decoded TokenData if the session has a valid JWT, else None."""
    token = st.session_state.get("jwt_token")
    if not token:
        return None
    return decode_token(token)


def _logout():
    for key in ("jwt_token", "auth_user", "db", "agent", "db_type",
                "current_df", "cleaned_df", "current_page",
                "last_question", "last_response"):
        st.session_state.pop(key, None)
    st.rerun()


def login_page():
    """Full-page login / signup UI shown when no valid JWT is in session."""
    st.markdown("""
        <style>
        .auth-container {
            max-width: 420px;
            margin: 60px auto 0 auto;
            padding: 2.5rem 2rem;
            background: #FFFFFF;
            border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
        }
        .auth-logo {
            text-align: center;
            font-size: 2.4rem;
            margin-bottom: 0.2rem;
        }
        .auth-title {
            text-align: center;
            font-size: 1.4rem;
            font-weight: 700;
            color: #1A1A2E;
            margin-bottom: 0.2rem;
        }
        .auth-subtitle {
            text-align: center;
            font-size: 0.85rem;
            color: #6B7280;
            margin-bottom: 1.5rem;
        }
        .auth-divider {
            border: none;
            border-top: 1px solid #E5E7EB;
            margin: 1.2rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown('<div class="auth-logo">📊</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-title">DataQuery AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">NL-to-SQL for Finance & Inventory</div>',
                    unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

        # ── Login tab ──────────────────────────────────────────────────────
        with tab_login:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="your_username")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    try:
                        payload = UserLogin(username=username, password=password)
                        user = authenticate_user(payload)
                        if user is None:
                            st.error("Incorrect username or password.")
                        else:
                            token = create_access_token(user)
                            st.session_state.jwt_token = token
                            st.session_state.auth_user = {
                                "user_id": user.user_id,
                                "username": user.username,
                                "email": user.email,
                            }
                            st.success(f"Welcome back, **{user.username}**!")
                            st.rerun()
                    except ValidationError as e:
                        st.error(e.errors()[0]["msg"])

        # ── Signup tab ─────────────────────────────────────────────────────
        with tab_signup:
            with st.form("signup_form", clear_on_submit=True):
                new_username = st.text_input("Username", placeholder="e.g. john_doe",
                                             key="su_username")
                new_email = st.text_input("Email", placeholder="you@example.com",
                                          key="su_email")
                new_password = st.text_input("Password", type="password",
                                             placeholder="Min 8 chars, 1 uppercase, 1 digit",
                                             key="su_password")
                confirm_password = st.text_input("Confirm Password", type="password",
                                                 placeholder="Repeat password",
                                                 key="su_confirm")
                st.caption("Password must be at least 8 characters with one uppercase letter and one digit.")
                submitted_signup = st.form_submit_button("Create Account",
                                                         use_container_width=True)

            if submitted_signup:
                try:
                    payload = UserCreate(
                        username=new_username,
                        email=new_email,
                        password=new_password,
                        confirm_password=confirm_password,
                    )
                    user = create_user(payload)
                    token = create_access_token(user)
                    st.session_state.jwt_token = token
                    st.session_state.auth_user = {
                        "user_id": user.user_id,
                        "username": user.username,
                        "email": user.email,
                    }
                    st.success(f"Account created! Welcome, **{user.username}**.")
                    st.rerun()
                except ValidationError as e:
                    for err in e.errors():
                        st.error(err["msg"])
                except ValueError as e:
                    st.error(str(e))


def _connect_database(db_type: str):
    """(Re)connect to the selected database backend and rebuild the agent.

    Connection failures (bad credentials, unreachable host, missing
    driver, etc.) are caught and stored in session state instead of
    raising - previously DatabaseConnection() was constructed once at
    import time with no error handling at all, so any connection problem
    crashed the whole app before it could even render a page.
    """
    try:
        st.session_state.db = DatabaseConnection(db_type=db_type)
        st.session_state.agent = EnhancedSQLAgent(st.session_state.db)
        st.session_state.db_type = db_type
        st.session_state.db_error = None
    except Exception as e:
        st.session_state.db = None
        st.session_state.agent = None
        st.session_state.db_error = str(e)


# Initialize session state
if 'db_type' not in st.session_state:
    default_db = Config.DEFAULT_DATABASE.lower() if Config.DEFAULT_DATABASE else 'sqlite'
    st.session_state.db_type = default_db if default_db in SUPPORTED_DB_BACKENDS else 'sqlite'

if 'db' not in st.session_state:
    _connect_database(st.session_state.db_type)

if 'current_df' not in st.session_state:
    st.session_state.current_df = None

if 'cleaned_df' not in st.session_state:
    st.session_state.cleaned_df = None

if 'current_page' not in st.session_state:
    st.session_state.current_page = "SQL Agent Chat"


def main():
    # ── Auth guard ─────────────────────────────────────────────────────────
    user_token = _current_user()
    if user_token is None:
        login_page()
        return

    user_id = user_token.user_id
    username = user_token.username

    st.title("📊 Universal SQL Database Agent for Finance & Inventory Datasets")
    st.markdown("Support for finance, sales, inventory, and supply chain datasets with automatic schema detection")

    # Sidebar
    with st.sidebar:
        # User info + logout
        st.markdown(f"**👤 {username}**")
        st.caption(f"User ID: {user_id}")
        if st.button("Sign Out", use_container_width=True):
            _logout()
        st.divider()
        st.header("Database")
        selected_db = st.selectbox(
            "Database backend",
            options=SUPPORTED_DB_BACKENDS,
            index=SUPPORTED_DB_BACKENDS.index(st.session_state.db_type),
            help="SQLite is built in and works on first launch. MySQL/PostgreSQL use the same SQLAlchemy pipeline."
        )
        if selected_db != st.session_state.db_type:
            with st.spinner(f"Connecting to {selected_db}..."):
                _connect_database(selected_db)

        if st.session_state.get('db_error'):
            st.error(f"⚠️ Database connection failed: {st.session_state.db_error}")
        elif st.session_state.get('db'):
            st.success(f"✅ Connected ({st.session_state.db_type})")
            st.caption(st.session_state.db.connection_label())

        with st.expander("Connection settings"):
            st.write("SQLite uses `SQLITE_DATABASE` and does not need a password.")
            st.write("For MySQL, put your password in `MYSQL_PASSWORD` inside `env` or `.env`.")
            st.write("For PostgreSQL, put your password in `POSTGRES_PASSWORD` inside `env` or `.env`.")
            if Config.GEMINI_API_KEY:
                st.write(f"Gemini key loaded: yes ({len(Config.GEMINI_API_KEY)} chars)")
            else:
                st.write("Gemini key loaded: no")

        if st.session_state.get('agent'):
            _render_active_dataset_selector()

        st.header("Navigation")
        pages = ["Data Upload & Cleaning", "SQL Agent Chat", "KPI Dashboard",
                 "Data Explorer", "Schema Analysis", "Inventory Analytics"]
        page = st.radio(
            "Choose a page:",
            pages,
            index=pages.index(st.session_state.current_page),
            key="nav_radio"
        )
        if page != st.session_state.current_page:
            st.session_state.current_page = page

    if st.session_state.get('db') is None:
        st.warning(
            "Not connected to a database. Check the connection settings for "
            f"**{st.session_state.db_type}** in your `.env` file, or pick a "
            "different backend in the sidebar."
        )
        return

    page = st.session_state.current_page
    if page == "Data Upload & Cleaning":
        data_upload_page()
    elif page == "SQL Agent Chat":
        agent_chat_page()
    elif page == "KPI Dashboard":
        kpi_dashboard_page()
    elif page == "Data Explorer":
        data_explorer_page()
    elif page == "Schema Analysis":
        schema_analysis_page()
    else:
        inventory_analytics_page()


def _render_active_dataset_selector():
    """Render one shared active-dataset selector — shows only current user's tables."""
    user_token = _current_user()
    if not user_token:
        return

    user_id = user_token.user_id
    all_tables = st.session_state.agent._get_available_tables()

    # Only tables that belong to this user
    user_tables = filter_user_tables(user_id, all_tables)
    if not user_tables:
        st.info("No datasets loaded yet. Upload a file to get started.")
        return

    # Show display names (without prefix) but store actual prefixed names
    display_names = [strip_user_prefix(user_id, t) for t in user_tables]
    table_display_map = dict(zip(display_names, user_tables))  # display → actual
    actual_display_map = dict(zip(user_tables, display_names))  # actual → display

    current_actual = st.session_state.agent.current_dataset
    current_display = actual_display_map.get(current_actual, display_names[0])

    if current_display not in display_names:
        current_display = display_names[0]

    selected_display = st.selectbox(
        "Active dataset",
        display_names,
        index=display_names.index(current_display),
        key="global_active_dataset"
    )
    selected_actual = table_display_map[selected_display]

    if selected_actual != st.session_state.agent.current_dataset:
        st.session_state.agent.set_current_dataset(selected_actual)
        st.session_state.last_question = ""
        st.session_state.last_response = ""
        st.rerun()

    st.caption(f"Current Dataset: **{selected_display}**")


def data_upload_page():
    st.header("📤 Data Upload & Cleaning")
    st.markdown("""
    Upload your dataset (CSV, Excel, or JSON). The system will automatically detect:
    - Schema and data types
    - Dataset type (Sales, Inventory, Customer, Transactions)
    - Column mappings
    - Suggested metrics

    **Supported Dataset Types:**
    - 📊 Sales & Orders
    - 📦 Inventory Management  
    - 👥 Customer Data
    - 💳 Transactions
    - 🏭 Supply Chain
    """)

    # File upload
    uploaded_file = st.file_uploader(
        "Upload your data file",
        type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
        help="Supported formats: CSV, Excel, JSON, Parquet"
    )

    if uploaded_file is not None:
        if uploaded_file.size > Config.MAX_FILE_SIZE:
            st.error(
                f"File is too large ({uploaded_file.size / (1024 * 1024):.1f} MB). "
                f"Maximum allowed size is {Config.MAX_FILE_SIZE / (1024 * 1024):.0f} MB."
            )
            return

        try:
            # Read file based on type
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            elif file_extension == 'parquet':
                df = pd.read_parquet(uploaded_file)
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return

            st.session_state.current_df = df

            # Display raw data
            st.subheader("📋 Raw Data Preview")
            st.dataframe(df.head(10))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

            # Schema detection
            st.subheader("🔍 Schema Detection")
            schema_detector = SchemaDetector()
            schema = schema_detector.detect_schema(df)

            dataset_type = schema.get('dataset_type', 'general').replace('_', ' ').title()
            is_inventory = schema.get('dataset_type') == 'inventory_management'

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Dataset Type", dataset_type)
            with col2:
                st.metric("Date Columns", len(schema.get('date_columns', [])))
            with col3:
                st.metric("Numeric Columns", len(schema.get('numeric_columns', [])))
            with col4:
                st.metric("String Columns", len(schema.get('string_columns', [])))

            # Show inventory-specific detection
            if is_inventory:
                st.info("📦 Inventory Dataset Detected!")
                inventory_cols = schema.get('inventory_columns', {})
                detected_cols = [col for col in inventory_cols.values() if col is not None]
                if detected_cols:
                    st.write("**Detected Inventory Columns:**")
                    for col_type, col_name in inventory_cols.items():
                        if col_name:
                            st.write(f"- {col_type.replace('_', ' ').title()}: {col_name}")

            with st.expander("View Detailed Schema"):
                st.json(schema)

            # Data cleaning
            st.subheader("🧹 Data Cleaning & Normalization")

            col1, col2 = st.columns(2)

            with col1:
                cleaning_level = st.selectbox(
                    "Cleaning Level:",
                    ["Basic (Remove duplicates, handle missing values)",
                     "Standard (Basic + normalize dates and strings)",
                     "Advanced (Standard + standardize categorical values, add metadata)"]
                )

            with col2:
                dataset_type_override = st.selectbox(
                    "Dataset Type (Override):",
                    ["Auto-detect", "Sales/Orders", "Inventory Management", "Transactions",
                     "Customer Data", "Product Data", "Supply Chain", "General"]
                )

            if st.button("Run Cleaning Pipeline", use_container_width=True):
                with st.spinner("Cleaning and normalizing data..."):
                    # Normalize data
                    normalizer = DataNormalizer()

                    # Override dataset type if specified
                    if dataset_type_override != "Auto-detect":
                        schema['dataset_type'] = dataset_type_override.lower().replace('/', '_')

                    cleaned_df = normalizer.normalize_dataframe(
                        df, schema, level=cleaning_level.split()[0].lower()
                    )
                    st.session_state.cleaned_df = cleaned_df

                    # Show results
                    st.subheader("✅ Cleaned Data Preview")
                    st.dataframe(cleaned_df.head(10))

                    st.success(f"Cleaning completed! Original: {len(df)} rows, Cleaned: {len(cleaned_df)} rows")

                    # Show cleaning summary
                    st.subheader("📊 Cleaning Summary")
                    cols = st.columns(4)

                    with cols[0]:
                        st.metric("Original Rows", len(df))
                    with cols[1]:
                        st.metric("Cleaned Rows", len(cleaned_df))
                    with cols[2]:
                        st.metric("Removed Rows", len(df) - len(cleaned_df))
                    with cols[3]:
                        st.metric("Columns Added", len(cleaned_df.columns) - len(df.columns))

                    # Show inventory-specific metrics if applicable
                    if schema.get('dataset_type') == 'inventory_management':
                        st.subheader("📦 Inventory Metrics")
                        inventory_metrics = schema.get('inventory_metrics', {})
                        if inventory_metrics:
                            metric_cols = st.columns(3)
                            for idx, (metric, value) in enumerate(inventory_metrics.items()):
                                if idx < 3:
                                    with metric_cols[idx % 3]:
                                        if isinstance(value, (int, float)):
                                            if 'value' in metric:
                                                st.metric(metric.replace('_', ' ').title(), f"${value:,.2f}")
                                            else:
                                                st.metric(metric.replace('_', ' ').title(), f"{value:,.0f}")
                                        else:
                                            st.metric(metric.replace('_', ' ').title(), value)

                    # Download cleaned data
                    csv = cleaned_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Cleaned Data",
                        data=csv,
                        file_name=f"cleaned_{uploaded_file.name}",
                        mime="text/csv"
                    )

            # Load to database
            if st.session_state.cleaned_df is not None:
                st.subheader("💾 Load to Database")
                base_table_name = st.text_input(
                    "Table Name:",
                    value=uploaded_file.name.split('.')[0].lower().replace(' ', '_')
                )

                # Scope the table to this user — stored as u{user_id}_{name}
                user_token = _current_user()
                if user_token:
                    actual_table_name = scoped_table_name(user_token.user_id, base_table_name)
                else:
                    actual_table_name = base_table_name

                st.caption(f"Will be stored as: `{actual_table_name}` (isolated to your account)")

                if st.button("Load to Database", use_container_width=True):
                    with st.spinner("Loading dataset into database..."):

                        success = st.session_state.agent.load_dataset(
                            st.session_state.cleaned_df,
                            actual_table_name,
                            already_normalized=True
                        )

                        if success:
                            st.session_state.agent.detect_schema(actual_table_name)
                            st.success(f"Dataset **{base_table_name}** loaded successfully.")
                            st.info(f"Rows Loaded: {len(st.session_state.cleaned_df)}")
                            st.rerun()

                        else:
                            st.error("Failed to load dataset.")
                            if getattr(st.session_state.agent, "last_error", None):
                                with st.expander("Show load error"):
                                    st.code(st.session_state.agent.last_error, language="text")
                            else:
                                st.write("See terminal for complete traceback.")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


def agent_chat_page():
    st.header("🤖 SQL Agent Chat")
    st.markdown("Ask questions about your finance or inventory data in natural language")

    # Sidebar with options
    with st.sidebar:
        st.subheader("Quick Actions")

        # Dataset info
        if st.session_state.agent.current_dataset:
            schema = st.session_state.agent.detect_schema()
            dataset_type = schema.get('dataset_type', 'unknown')
            st.info(f"Current Dataset: {st.session_state.agent.current_dataset}")
            st.info(f"Dataset Type: {dataset_type.replace('_', ' ').title()}")

        # Predefined questions by category
        st.write("**Sales Questions:**")
        sales_questions = [
            "What are the top 5 customers by revenue?",
            "Show me total sales by region",
            "What is the total profit for each category?",
            "Show me monthly sales trend for the last year"
        ]

        for i, q in enumerate(sales_questions):
            if st.button(
                    q,
                    use_container_width=True,
                    key=f"sales_question_{i}"
            ):
                st.session_state.last_question = q
                with st.spinner("Processing..."):
                    response = st.session_state.agent.answer_question(q)
                st.session_state.last_response = response
                st.rerun()

        st.divider()

        st.write("**Inventory Questions:**")
        inventory_questions = [
            "What is the current stock status?",
            "Which items are low in stock?",
            "What is the total inventory value?",
            "Show me items that need reordering"
        ]

        for i, q in enumerate(inventory_questions):
            if st.button(
                    q,
                    use_container_width=True,
                    key=f"inventory_question_{i}"
            ):
                st.session_state.last_question = q
                with st.spinner("Processing..."):
                    response = st.session_state.agent.answer_question(q)
                st.session_state.last_response = response
                st.rerun()

        st.divider()

        # KPI shortcuts
        st.write("**KPI Shortcuts:**")
        kpis = st.session_state.agent.get_kpi_list()
        for i, kpi in enumerate(kpis[:5]):
            if st.button(
                    f"📊 {kpi}",
                    use_container_width=True,
                    key=f"kpi_{i}"
            ):
                st.session_state.last_question = kpi
                with st.spinner("Processing..."):
                    response = st.session_state.agent.answer_question(kpi)
                st.session_state.last_response = response
                st.rerun()

    # Chat interface
    st.subheader("💬 Ask Questions")

    # Display last response
    if 'last_response' in st.session_state:
        st.info(f"**Question:** {st.session_state.get('last_question', '')}")
        st.write(st.session_state.last_response)

        # Try to visualize if it's a table
        if '|' in st.session_state.last_response and '\n' in st.session_state.last_response:
            try:
                lines = [line.strip() for line in st.session_state.last_response.split('\n') if line.strip()]
                if len(lines) > 1:
                    # Find header and data
                    for i, line in enumerate(lines):
                        if '|' in line and '-' not in line:
                            headers = [h.strip() for h in line.split('|') if h.strip()]
                            data_lines = []
                            for j in range(i + 1, len(lines)):
                                if '|' in lines[j] and '-' not in lines[j]:
                                    data_lines.append([d.strip() for d in lines[j].split('|') if d.strip()])

                            if data_lines and len(data_lines) > 1:
                                df = pd.DataFrame(data_lines, columns=headers)
                                if len(df) > 0 and len(df.columns) > 1:
                                    # Try to create visualization
                                    st.subheader("📊 Visualization")

                                    # Determine numeric columns
                                    numeric_cols = []
                                    for col in df.columns:
                                        try:
                                            pd.to_numeric(df[col])
                                            numeric_cols.append(col)
                                        except:
                                            pass

                                    if len(numeric_cols) >= 1:
                                        # Bar chart
                                        fig = px.bar(df, x=df.columns[0], y=numeric_cols[0],
                                                     title=f"{numeric_cols[0]} by {df.columns[0]}")
                                        st.plotly_chart(fig, use_container_width=True)

                                        # Additional charts
                                        if len(numeric_cols) >= 2:
                                            # Scatter plot
                                            fig2 = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                                                              text=df.columns[0],
                                                              title=f"{numeric_cols[1]} vs {numeric_cols[0]}")
                                            st.plotly_chart(fig2, use_container_width=True)

                                            # Pie chart for distribution
                                            if len(df) <= 20:
                                                fig3 = px.pie(df, values=numeric_cols[0], names=df.columns[0],
                                                              title=f"{numeric_cols[0]} Distribution")
                                                st.plotly_chart(fig3, use_container_width=True)
            except:
                pass

    # Input for custom question
    st.divider()
    question = st.text_input(
        "Type your question:",
        placeholder="e.g., What is the current stock status?",
        key="question_input"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_clicked = st.button("Ask", use_container_width=True)

    if ask_clicked and question:
        with st.spinner("Processing your question..."):
            response = st.session_state.agent.answer_question(question)
        st.session_state.last_question = question
        st.session_state.last_response = response
        st.rerun()

    # Show SQL suggestion
    with col2:
        if st.button("Show SQL Query", use_container_width=True):
            if question:
                sql = st.session_state.agent.get_query_suggestions(question)
                st.code(sql, language="sql")


def inventory_analytics_page():
    st.header("📦 Inventory Analytics Dashboard")
    st.markdown("Comprehensive inventory management analytics and insights")

    # Check if dataset is inventory
    schema = st.session_state.agent.detect_schema()
    dataset_type = schema.get('dataset_type', 'general')

    if dataset_type != 'inventory_management':
        st.warning("This page is designed for inventory datasets. Please upload an inventory dataset first.")
        st.info("Expected inventory columns: stock_level, reorder_point, supplier, warehouse, turnover, etc.")
        return

    # Inventory Overview
    st.subheader("📊 Inventory Overview")

    col1, col2, col3, col4 = st.columns(4)

    # Get key metrics
    try:
        total_stock = st.session_state.agent.execute_kpi_query("Total Stock Quantity")
        total_value = st.session_state.agent.execute_kpi_query("Total Inventory Value")
        low_stock = st.session_state.agent.execute_kpi_query("Low Stock Items")
        out_of_stock = st.session_state.agent.execute_kpi_query("Out of Stock Items")

        with col1:
            st.metric("Total Stock Quantity", total_stock.split('\n')[1].strip() if '\n' in total_stock else "N/A")
        with col2:
            st.metric("Total Inventory Value", total_value.split('\n')[1].strip() if '\n' in total_value else "N/A")
        with col3:
            st.metric("Low Stock Items", low_stock.split('\n')[1].strip() if '\n' in low_stock else "N/A")
        with col4:
            st.metric("Out of Stock Items", out_of_stock.split('\n')[1].strip() if '\n' in out_of_stock else "N/A")
    except:
        st.warning("Could not load inventory metrics. Please ensure inventory columns are properly mapped.")

    st.divider()

    # Analysis Options
    st.subheader("📈 Inventory Analysis")

    analysis_types = [
        "Stock Status Distribution",
        "Inventory Value by Category",
        "Stock Turnover Analysis",
        "Reorder Point Analysis",
        "Supplier Performance",
        "Warehouse Distribution",
        "Expiry Date Analysis",
        "ABC Classification"
    ]

    selected_analysis = st.selectbox("Select Analysis Type:", analysis_types)

    if st.button("Run Analysis", use_container_width=True):
        with st.spinner("Running analysis..."):
            if "Stock Status" in selected_analysis:
                result = st.session_state.agent.inventory_analysis('stock_status')
            elif "Inventory Value" in selected_analysis:
                result = st.session_state.agent.inventory_analysis('inventory_value')
            elif "Turnover" in selected_analysis:
                result = st.session_state.agent.inventory_analysis('turnover_analysis')
            elif "Reorder" in selected_analysis:
                result = st.session_state.agent.inventory_analysis('reorder_analysis')
            elif "Supplier" in selected_analysis:
                result = st.session_state.agent.inventory_analysis('supplier_analysis')
            elif "Warehouse" in selected_analysis:
                result = st.session_state.agent.inventory_analysis('warehouse_analysis')
            elif "Expiry" in selected_analysis:
                result = st.session_state.agent.inventory_analysis('expiry_analysis')
            elif "ABC" in selected_analysis:
                criteria = 'value' if 'value' in selected_analysis.lower() else 'quantity'
                result = st.session_state.agent.abc_analysis(criteria)
            else:
                result = st.session_state.agent.inventory_analysis('stock_status')

            if result is not None and not result.empty:
                st.subheader("📊 Analysis Results")

                if list(result.columns) == ['message']:
                    st.info(result['message'].iloc[0])
                else:
                    st.dataframe(result, use_container_width=True)

                    numeric_cols = [
                        col for col in result.columns
                        if pd.api.types.is_numeric_dtype(result[col])
                    ]

                    if numeric_cols and len(result.columns) >= 2:
                        # Bar chart
                        fig = px.bar(result, x=result.columns[0], y=numeric_cols[0],
                                     title=f"{numeric_cols[0]} Analysis")
                        st.plotly_chart(fig, use_container_width=True)

                        # Additional charts based on analysis type
                        if "Stock Status" in selected_analysis:
                            fig2 = px.pie(result, values=numeric_cols[0], names=result.columns[0],
                                          title="Stock Status Distribution")
                            st.plotly_chart(fig2, use_container_width=True)

                        if "Turnover" in selected_analysis:
                            fig3 = px.scatter(result, x=result.columns[0], y=numeric_cols[0],
                                              size=numeric_cols[0] if len(numeric_cols) > 1 else None,
                                              title="Turnover Analysis")
                            st.plotly_chart(fig3, use_container_width=True)

                        if "ABC" in selected_analysis and 'ABC_Class' in result.columns:
                            colors = {'A': '#00FF00', 'B': '#FFFF00', 'C': '#FF0000'}
                            fig4 = px.bar(result, x='ABC_Class', y=numeric_cols[0],
                                          color='ABC_Class', color_discrete_map=colors,
                                          title="ABC Classification Distribution")
                            st.plotly_chart(fig4, use_container_width=True)

                    csv = result.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"{selected_analysis.lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No data found for this analysis")


def _format_kpi_metric_value(result_df: pd.DataFrame, metric: Dict) -> str:
    """Format scalar KPI values for st.metric without truncating tables."""
    if result_df is None or isinstance(result_df, str) or result_df.empty:
        return "N/A"

    metric_type = metric.get('type')

    if metric_type == 'date_range' and len(result_df.columns) >= 2:
        start = str(result_df.iloc[0, 0]).split()[0]
        end = str(result_df.iloc[0, 1]).split()[0]
        return f"{start} to {end}"

    value = result_df.iloc[0, 0]
    if pd.isna(value):
        return "N/A"

    metric_name = (metric.get('name') or '').lower()
    if isinstance(value, (int, float)):
        if any(word in metric_name for word in ['revenue', 'profit', 'sales', 'value']):
            return f"${value:,.2f}"
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}"

    return str(value)


def kpi_dashboard_page():
    st.header("📈 KPI Dashboard")
    st.markdown("Monitor key performance indicators for your finance or inventory data")

    # Get KPIs
    kpis = st.session_state.agent.get_kpi_list()

    if not kpis:
        st.warning("No KPIs available. Please load a dataset first.")
        return

    schema = st.session_state.agent.detect_schema()
    metrics = schema.get('suggested_metrics', [])
    scalar_metrics = []
    detail_kpis = []
    for metric in metrics:
        if metric.get('type') in ('numeric', 'count', 'date_range'):
            scalar_metrics.append(metric)
        else:
            detail_kpis.append(metric.get('name'))

    # Display compact scalar KPIs only. Table/time-series metrics belong in
    # Quick Analysis, not squeezed into st.metric cards.
    cols = st.columns(4)
    for idx, metric in enumerate(scalar_metrics[:8]):
        with cols[idx % 4]:
            kpi = metric.get('name')
            with st.spinner(f"Loading {kpi}..."):
                try:
                    result_df = st.session_state.agent.execute_kpi_dataframe(kpi)
                    value = _format_kpi_metric_value(result_df, metric)
                    st.metric(kpi, value)
                except:
                    st.metric(kpi, "Error")

    if detail_kpis:
        st.caption("Detailed customer and time-series KPIs are available below in Quick Analysis.")

    st.divider()

    # Quick analysis
    st.subheader("📊 Quick Analysis")

    # Get available analysis types based on dataset
    dataset_type = schema.get('dataset_type', 'general')

    if dataset_type == 'inventory_management':
        analysis_options = [
            "Stock Status",
            "Inventory Value",
            "Stock Turnover",
            "Supplier Analysis",
            "Warehouse Analysis",
            "Expiry Analysis",
            "ABC Classification"
        ]
    else:
        analysis_options = [
            "Sales by Region",
            "Sales by Category",
            "Monthly Sales Trend",
            "Monthly Profit Trend",
            "YoY Revenue",
            "MoM Revenue",
            "Top Products",
            "Customer Analysis",
            "Profit Analysis"
        ]

    analysis_type = st.selectbox("Select Analysis:", analysis_options)

    if st.button("Run Analysis", use_container_width=True):
        with st.spinner("Running analysis..."):
            # Generate appropriate query based on analysis type
            if dataset_type == 'inventory_management':
                if "Turnover" in analysis_type:
                    result = st.session_state.agent.inventory_analysis('turnover_analysis')
                elif "ABC" in analysis_type:
                    result = st.session_state.agent.abc_analysis('value')
                else:
                    result = st.session_state.agent.inventory_analysis(analysis_type.lower().replace(' ', '_'))
            else:
                query = generate_analysis_query(analysis_type, schema, st.session_state.db.db_type)
                result = st.session_state.db.execute_query(query)

            if result is not None and not result.empty:
                st.subheader(f"📊 {analysis_type}")

                if list(result.columns) == ['message']:
                    st.info(result['message'].iloc[0])
                else:
                    st.dataframe(result)

                    # Create visualizations
                    if len(result.columns) >= 2:
                        # Determine chart type based on data
                        numeric_cols = result.select_dtypes(include=['number']).columns

                        if len(numeric_cols) >= 1:
                            # Bar chart
                            fig = px.bar(result, x=result.columns[0], y=numeric_cols[0],
                                         title=f"{numeric_cols[0]} Analysis")
                            st.plotly_chart(fig, use_container_width=True)

                            # Additional charts
                            if len(result.columns) == 2 and len(result) <= 20:
                                # Pie chart
                                fig2 = px.pie(result, values=result.columns[1], names=result.columns[0],
                                              title=f"{result.columns[1]} Distribution")
                                st.plotly_chart(fig2, use_container_width=True)

                            if len(numeric_cols) >= 2:
                                # Line chart for trends
                                fig3 = px.line(result, x=result.columns[0], y=numeric_cols,
                                               title="Trend Analysis")
                                st.plotly_chart(fig3, use_container_width=True)

                    # Download
                    csv = result.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Data",
                        data=csv,
                        file_name=f"{analysis_type.lower().replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No data found for this analysis")


def generate_analysis_query(analysis_type: str, schema: Dict, db_type: str = "sqlite") -> str:
    """Generate SQL query based on analysis type"""
    mapping = schema.get('column_mapping', {})
    table = schema.get('table_name', '')

    # If inventory dataset, use inventory analysis
    if schema.get('dataset_type') == 'inventory_management':
        return f"SELECT * FROM {table} LIMIT 10"

    order_date = mapping.get('order_date') or mapping.get('date') or 'order_date'
    sales_col = mapping.get('sales', 'sales')
    profit_col = mapping.get('profit', 'profit')
    region_col = mapping.get('region', 'region')
    category_col = mapping.get('category', 'category')
    product_col = mapping.get('product_name', 'product_name')
    customer_id_col = mapping.get('customer_id', 'customer_id')
    customer_name_col = mapping.get('customer_name', 'customer_name')

    if db_type == "mysql":
        month_expr = f"DATE_FORMAT({order_date}, '%Y-%m')"
        year_expr = f"YEAR({order_date})"
    elif db_type == "postgresql":
        month_expr = f"TO_CHAR(DATE_TRUNC('month', {order_date}), 'YYYY-MM')"
        year_expr = f"TO_CHAR(DATE_TRUNC('year', {order_date}), 'YYYY')"
    else:
        month_expr = f"strftime('%Y-%m', {order_date})"
        year_expr = f"strftime('%Y', {order_date})"

    # Sales analysis queries
    analysis_queries = {
        "Sales by Region": f"""
            SELECT {region_col}, 
                   ROUND(SUM({sales_col}), 2) as Total_Sales
            FROM {table}
            GROUP BY {region_col}
            ORDER BY Total_Sales DESC
        """,
        "Sales by Category": f"""
            SELECT {category_col}, 
                   ROUND(SUM({sales_col}), 2) as Total_Sales
            FROM {table}
            GROUP BY {category_col}
            ORDER BY Total_Sales DESC
        """,
        "Monthly Sales Trend": f"""
            SELECT {month_expr} as Month,
                   ROUND(SUM({sales_col}), 2) as Total_Sales
            FROM {table}
            WHERE {order_date} IS NOT NULL
            GROUP BY {month_expr}
            ORDER BY Month
        """,
        "Monthly Profit Trend": f"""
            SELECT {month_expr} as Month,
                   ROUND(SUM({profit_col}), 2) as Total_Profit
            FROM {table}
            WHERE {order_date} IS NOT NULL
            GROUP BY {month_expr}
            ORDER BY Month
        """,
        "YoY Revenue": f"""
            WITH yearly AS (
                SELECT {year_expr} as Year,
                       ROUND(SUM({sales_col}), 2) as Revenue,
                       ROUND(SUM({profit_col}), 2) as Profit
                FROM {table}
                WHERE {order_date} IS NOT NULL
                GROUP BY {year_expr}
            )
            SELECT Year,
                   Revenue,
                   Profit,
                   ROUND(Revenue - LAG(Revenue) OVER (ORDER BY Year), 2) as YoY_Revenue_Change,
                   ROUND(
                       (Revenue - LAG(Revenue) OVER (ORDER BY Year))
                       / NULLIF(LAG(Revenue) OVER (ORDER BY Year), 0) * 100,
                       2
                   ) as YoY_Revenue_Change_Pct
            FROM yearly
            ORDER BY Year
        """,
        "MoM Revenue": f"""
            WITH monthly AS (
                SELECT {month_expr} as Month,
                       ROUND(SUM({sales_col}), 2) as Revenue,
                       ROUND(SUM({profit_col}), 2) as Profit
                FROM {table}
                WHERE {order_date} IS NOT NULL
                GROUP BY {month_expr}
            )
            SELECT Month,
                   Revenue,
                   Profit,
                   ROUND(Revenue - LAG(Revenue) OVER (ORDER BY Month), 2) as MoM_Revenue_Change,
                   ROUND(
                       (Revenue - LAG(Revenue) OVER (ORDER BY Month))
                       / NULLIF(LAG(Revenue) OVER (ORDER BY Month), 0) * 100,
                       2
                   ) as MoM_Revenue_Change_Pct
            FROM monthly
            ORDER BY Month
        """,
        "Top Products": f"""
            SELECT {product_col} as Product,
                   ROUND(SUM({sales_col}), 2) as Total_Sales,
                   ROUND(SUM({profit_col}), 2) as Total_Profit
            FROM {table}
            GROUP BY {product_col}
            ORDER BY Total_Sales DESC
            LIMIT 10
        """,
        "Customer Analysis": f"""
            SELECT {customer_id_col} as Customer_ID,
                   {customer_name_col} as Customer_Name,
                   ROUND(SUM({sales_col}), 2) as Total_Revenue,
                   ROUND(SUM({profit_col}), 2) as Total_Profit,
                   COUNT(DISTINCT {mapping.get('order_id', customer_id_col)}) as Order_Count,
                   ROUND(SUM({sales_col}) / NULLIF(COUNT(DISTINCT {mapping.get('order_id', customer_id_col)}), 0), 2) as Avg_Order_Value
            FROM {table}
            GROUP BY {customer_id_col}, {customer_name_col}
            ORDER BY Total_Revenue DESC
            LIMIT 25
        """,
        "Profit Analysis": f"""
            SELECT {category_col} as Category,
                   ROUND(SUM({sales_col}), 2) as Total_Sales,
                   ROUND(SUM({profit_col}), 2) as Total_Profit,
                   ROUND(SUM({profit_col}) / NULLIF(SUM({sales_col}), 0) * 100, 2) as Profit_Margin_Pct
            FROM {table}
            GROUP BY {category_col}
            ORDER BY Total_Profit DESC
        """
    }

    return analysis_queries.get(analysis_type, f"SELECT * FROM {table} LIMIT 10")


def data_explorer_page():
    st.header("🔍 Data Explorer")
    st.markdown("Explore your data with custom queries and filters")

    # Get list of tables
    tables = st.session_state.agent._get_available_tables()
    st.write("Detected Tables:", tables)
    if not tables:
        st.warning("No tables available. Please load a dataset first.")
        return

    default_index = 0
    if st.session_state.agent.current_dataset in tables:
        default_index = tables.index(st.session_state.agent.current_dataset)

    selected_table = st.selectbox("Select Table:", tables, index=default_index)
    if selected_table and selected_table != st.session_state.agent.current_dataset:
        st.session_state.agent.set_current_dataset(selected_table)

    if selected_table:
        # Get schema
        schema = st.session_state.db.get_table_schema(selected_table)
        if not schema.empty:
            st.subheader(f"📋 {selected_table} Schema")
            st.dataframe(schema)

        # Get sample data
        sample = st.session_state.db.get_sample_data(selected_table)
        if not sample.empty:
            st.subheader(f"📊 {selected_table} Sample Data")
            st.dataframe(sample)

            # Row count
            count_query = f"SELECT COUNT(*) as row_count FROM {selected_table}"
            count_df = st.session_state.db.execute_query(count_query)
            if not count_df.empty:
                st.metric("Total Rows", count_df.iloc[0, 0])

        # Custom query
        st.divider()
        st.subheader("🔎 Custom Query")

        # Query builder
        with st.expander("Query Builder"):
            col1, col2 = st.columns(2)

            with col1:
                # Column selection
                columns = sample.columns.tolist() if not sample.empty else []
                selected_cols = st.multiselect("Select Columns:", columns, default=columns[:3])

                # Limit
                limit = st.number_input("Limit:", min_value=1, max_value=10000, value=10)

            with col2:
                # Filter conditions
                st.write("Filter Conditions:")
                filter_col = st.selectbox("Column:", columns, key="filter_col")
                filter_op = st.selectbox("Operator:", ["=", ">", "<", ">=", "<=", "!=", "LIKE"])
                filter_val = st.text_input("Value:", "")

                # Sort
                sort_col = st.selectbox("Sort By:", columns, key="sort_col")
                sort_order = st.selectbox("Order:", ["ASC", "DESC"])

        # Generate query
        query = f"SELECT {', '.join(selected_cols) if selected_cols else '*'} FROM {selected_table}"
        display_query = query
        params = {}

        if filter_col and filter_val:
            if filter_op == "LIKE":
                query += f" WHERE {filter_col} LIKE :filter_val"
                params['filter_val'] = f"%{filter_val}%"
                display_query += f" WHERE {filter_col} LIKE '%{filter_val}%'"
            else:
                query += f" WHERE {filter_col} {filter_op} :filter_val"
                params['filter_val'] = filter_val
                display_query += f" WHERE {filter_col} {filter_op} '{filter_val}'"

        if sort_col:
            query += f" ORDER BY {sort_col} {sort_order}"
            display_query += f" ORDER BY {sort_col} {sort_order}"

        query += f" LIMIT {limit}"
        display_query += f" LIMIT {limit}"

        st.code(display_query, language="sql")

        if st.button("Execute Query", use_container_width=True):
            with st.spinner("Executing query..."):
                result = st.session_state.db.execute_query(query, params=params)
                if not result.empty:
                    st.dataframe(result)
                    st.write(f"**Rows:** {len(result)}")

                    # Download
                    csv = result.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="query_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No results or query executed successfully")


def schema_analysis_page():
    st.header("🔍 Schema Analysis")
    st.markdown("Analyze and understand your dataset schema in detail")

    # Get database tables
    tables = st.session_state.agent._get_available_tables()
    if not tables:
        st.warning("No datasets loaded. Please upload and load a dataset first.")
        return

    default_index = 0
    if st.session_state.agent.current_dataset in tables:
        default_index = tables.index(st.session_state.agent.current_dataset)

    selected_dataset = st.selectbox(
        "Select Dataset:",
        tables,
        index=default_index
    )

    if selected_dataset:
        schema = st.session_state.agent.set_current_dataset(selected_dataset)

        # Overview
        st.subheader("📊 Dataset Overview")
        dataset_type = schema.get('dataset_type', 'unknown')
        is_inventory = dataset_type == 'inventory_management'

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Dataset Type", dataset_type.replace('_', ' ').title())
        with col2:
            st.metric("Total Columns", len(schema.get('columns', {})))
        with col3:
            st.metric("Date Columns", len(schema.get('date_columns', [])))
        with col4:
            st.metric("Numeric Columns", len(schema.get('numeric_columns', [])))

        # Inventory-specific section
        if is_inventory:
            st.subheader("📦 Inventory Columns")
            inventory_cols = schema.get('inventory_columns', {})
            if inventory_cols:
                inv_df = pd.DataFrame({
                    'Column Type': [k.replace('_', ' ').title() for k in inventory_cols.keys()],
                    'Actual Column Name': [v for v in inventory_cols.values()]
                })
                st.dataframe(inv_df)

            # Inventory metrics
            inventory_metrics = schema.get('inventory_metrics', {})
            if inventory_metrics:
                st.subheader("📊 Inventory Metrics")
                metric_cols = st.columns(4)
                for idx, (metric, value) in enumerate(inventory_metrics.items()):
                    if idx < 4:
                        with metric_cols[idx]:
                            if isinstance(value, (int, float)):
                                if 'value' in metric:
                                    st.metric(metric.replace('_', ' ').title(), f"${value:,.2f}")
                                else:
                                    st.metric(metric.replace('_', ' ').title(), f"{value:,.0f}")
                            else:
                                st.metric(metric.replace('_', ' ').title(), str(value))

        # Column analysis
        st.subheader("📋 Column Analysis")

        # Create DataFrame for column analysis
        col_data = []
        for col, info in schema.get('columns', {}).items():
            col_data.append({
                'Column Name': col,
                'Data Type': info.get('data_type', 'unknown'),
                'Null Count': info.get('null_count', 0),
                'Null %': f"{info.get('null_percentage', 0):.1f}%",
                'Unique Values': info.get('unique_count', 0),
                'Is ID': 'Yes' if info.get('is_id', False) else 'No',
                'Inventory Related': 'Yes' if info.get('is_inventory_related', False) else 'No',
                'Sample Values': str(info.get('sample_values', [])[:3])
            })

        col_df = pd.DataFrame(col_data)
        st.dataframe(col_df)

        # Column mapping
        st.subheader("🏷️ Column Mapping")
        mapping = schema.get('column_mapping', {})
        if mapping:
            mapping_df = pd.DataFrame({
                'Standard Name': list(mapping.keys()),
                'Actual Column': list(mapping.values())
            })
            st.dataframe(mapping_df)
        else:
            st.info("No column mappings identified")

        # Suggested metrics
        st.subheader("📈 Suggested Metrics")
        metrics = schema.get('suggested_metrics', [])
        if metrics:
            metrics_df = pd.DataFrame(metrics)
            st.dataframe(metrics_df)
        else:
            st.info("No metrics suggested for this dataset")

        # Data quality report
        st.subheader("✅ Data Quality Report")

        # Missing values analysis
        source_columns = {
            col for col in schema.get('columns', {}).keys()
            if not (
                col.startswith('_')
                or re.search(r'_(year|month|day|quarter|weekday|week|month_name|days_until)$', col)
            )
        }
        null_counts = {
            col: count for col, count in schema.get('null_counts', {}).items()
            if col in source_columns
        }
        if null_counts:
            null_df = pd.DataFrame({
                'Column': list(null_counts.keys()),
                'Missing Values': list(null_counts.values())
            })
            null_df = null_df[null_df['Missing Values'] > 0]

            if not null_df.empty:
                st.warning(f"{len(null_df)} columns have missing values")
                st.dataframe(null_df)

                # Bar chart of missing values
                fig = px.bar(null_df, x='Column', y='Missing Values',
                             title='Missing Values by Column')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values detected in the dataset")

        # Export schema
        st.download_button(
            label="📥 Export Schema as JSON",
            data=json.dumps(schema, indent=2, default=str),
            file_name="schema_analysis.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()