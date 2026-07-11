# config.py
import os
from dotenv import load_dotenv

# Load environment variables from a standard .env file, and also from the
# existing project-local "env" file for backwards compatibility.
load_dotenv()
load_dotenv("env", override=True)


class Config:
    # ==================================================
    # Database Configuration
    # ==================================================

    DEFAULT_DATABASE = os.getenv(
        "DEFAULT_DATABASE",
        "sqlite"
    )

    # -----------------------
    # MySQL
    # -----------------------

    MYSQL_HOST = os.getenv(
        "MYSQL_HOST",
        "localhost"
    )

    MYSQL_PORT = os.getenv(
        "MYSQL_PORT",
        "3306"
    )

    MYSQL_USER = os.getenv(
        "MYSQL_USER",
        "root"
    )

    MYSQL_PASSWORD = os.getenv(
        "MYSQL_PASSWORD",
        ""
    )

    MYSQL_DATABASE = os.getenv(
        "MYSQL_DATABASE",
        "inventory_db"
    )

    # -----------------------
    # PostgreSQL
    # -----------------------

    POSTGRES_HOST = os.getenv(
        "POSTGRES_HOST",
        "localhost"
    )

    POSTGRES_PORT = os.getenv(
        "POSTGRES_PORT",
        "5432"
    )

    POSTGRES_USER = os.getenv(
        "POSTGRES_USER",
        "postgres"
    )

    POSTGRES_PASSWORD = os.getenv(
        "POSTGRES_PASSWORD",
        ""
    )

    POSTGRES_DATABASE = os.getenv(
        "POSTGRES_DATABASE",
        "inventory_db"
    )

    # -----------------------
    # MongoDB
    # -----------------------

    MONGODB_URI = os.getenv(
        "MONGODB_URI",
        "mongodb://localhost:27017"
    )

    MONGODB_DATABASE = os.getenv(
        "MONGODB_DATABASE",
        "inventory_db"
    )

    # -----------------------
    # SQLite
    # -----------------------

    SQLITE_DATABASE = os.getenv(
        "SQLITE_DATABASE",
        "uploaded_dataset.db"
    )

    # ==================================================
    # Gemini Configuration
    # ==================================================
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '').strip().strip('"').strip("'")

    DEFAULT_MODEL = os.getenv(
        'DEFAULT_MODEL',
        'gemini-2.5-flash'
    )

    TEMPERATURE = float(
        os.getenv(
            'TEMPERATURE',
            '0'
        )
    )

    MAX_OUTPUT_TOKENS = int(
        os.getenv(
            'MAX_OUTPUT_TOKENS',
            '8192'
        )
    )

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    DEFAULT_PROVIDER = os.getenv(
        "DEFAULT_PROVIDER",
        "gemini"
    )

    FALLBACK_PROVIDER = os.getenv(
        "FALLBACK_PROVIDER",
        "groq"
    )

    GROQ_MODEL = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile"
    )

    # File Upload Configuration
    UPLOAD_DIR = os.getenv('UPLOAD_DIR', 'uploads')
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'parquet'}
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

    # Query Configuration

    MAX_QUERY_ROWS = 1000

    QUERY_TIMEOUT = 60

    # Column Mappings for Schema Detection
    COLUMN_MAPPINGS = {
        'date': ['date', 'created_date', 'transaction_date', 'invoice_date'],
        'order_date': ['order_date', 'order date', 'order_dt', 'order created date'],
        'ship_date': ['ship_date', 'ship date', 'shipping_date', 'dispatch_date'],
        'transaction_date': ['transaction_date', 'transaction date', 'payment_date', 'invoice_date'],
        'order_id': ['order_id', 'order_number', 'order_no', 'invoice_id', 'transaction_id'],
        'customer_id': ['customer_id', 'customer_number', 'client_id', 'user_id'],
        'product_id': ['product_id', 'product_code', 'item_id', 'sku', 'product_number'],
        'product_name': ['product_name', 'item_name', 'product', 'description', 'item_description'],
        'category': ['category', 'product_category', 'item_category', 'department', 'class'],
        'sub_category': ['sub_category', 'subcategory', 'subcat'],
        'sales': ['sales', 'revenue', 'amount', 'total', 'net_sales', 'gross_sales'],
        'profit': ['profit', 'net_profit', 'gross_profit', 'margin'],
        'quantity': ['quantity', 'qty', 'units', 'count', 'volume'],
        'price': ['price', 'unit_price', 'list_price', 'selling_price', 'rate'],
        'cost': ['cost', 'unit_cost', 'product_cost', 'cogs', 'wholesale_price'],
        'discount': ['discount', 'discount_rate', 'discount_percentage', 'disc'],
        # NOTE: region/country/state/city used to be merged into a single
        # 'region' bucket. That meant on datasets with several of these
        # columns at once (e.g. a Superstore-style dataset with Country,
        # State, City AND Region columns) only one would ever get mapped
        # and the rest were silently dropped. They're now separate.
        'region': ['region', 'territory', 'zone'],
        'country': ['country', 'nation'],
        'state': ['state', 'province'],
        'city': ['city', 'town'],
        'ship_mode': ['ship_mode', 'shipping_mode', 'delivery_mode', 'carrier'],
        'segment': ['segment', 'customer_segment', 'market_segment', 'group'],
        'customer_name': ['customer_name', 'client_name', 'company', 'business_name'],
        'supplier': ['supplier', 'vendor', 'manufacturer', 'brand', 'distributor']
    }

    DATE_FORMATS = [
        '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%Y%m%d', '%d-%m-%Y',
        '%b %d, %Y', '%B %d, %Y', '%Y-%m-%d %H:%M:%S.%f'
    ]

    # ==================================================
    # SQL Validation
    # ==================================================

    FORBIDDEN_SQL = [
        "DROP",
        "DELETE",
        "UPDATE",
        "ALTER",
        "TRUNCATE",
        "INSERT",
        "EXEC",
        "MERGE",
        "REPLACE"
    ]
