# schemas_detector.py
import pandas as pd
from typing import Dict, List, Any
import re
from datetime import datetime
import json
import warnings
from config import Config


class SchemaDetector:
    def __init__(self):
        self.column_mappings = Config.COLUMN_MAPPINGS
        self.date_formats = Config.DATE_FORMATS
        self.inventory_keywords = [
            'stock', 'inventory', 'warehouse', 'supply', 'batch', 'lot',
            'reorder', 'turnover', 'shelf', 'bin', 'pallet', 'storage'
        ]

    def detect_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect the schema and data types of a DataFrame"""
        schema = {
            'table_name': None,
            'columns': {},
            'primary_keys': [],
            'foreign_keys': [],
            'data_types': {},
            'date_columns': [],
            'numeric_columns': [],
            'string_columns': [],
            'id_columns': [],
            'unique_columns': [],
            'null_counts': {},
            'column_mapping': {},
            'suggested_metrics': [],
            'dataset_type': None,
            'inventory_columns': {},
            'inventory_metrics': {},
            'quality_metrics': {},
            'row_count': len(df)
        }

        # Analyze each column
        for col in df.columns:
            col_info = self._analyze_column(df, col)
            schema['columns'][col] = col_info

            # Categorize columns
            if col_info['data_type'] == 'date':
                schema['date_columns'].append(col)
            elif col_info['data_type'] in ['int', 'float', 'decimal', 'numeric']:
                schema['numeric_columns'].append(col)
            elif col_info['data_type'] in ['string', 'categorical']:
                schema['string_columns'].append(col)

            schema['data_types'][col] = col_info['data_type']

            # Detect ID columns
            if col_info['is_id']:
                schema['id_columns'].append(col)

            # Check for uniqueness
            if col_info['is_unique']:
                schema['unique_columns'].append(col)

            # Store null counts
            schema['null_counts'][col] = col_info['null_count']

        # Detect dataset type (including inventory)
        schema['dataset_type'] = self._detect_dataset_type(schema, df)

        # Generate column mappings
        schema['column_mapping'] = self._map_columns(schema)

        # Detect inventory-specific columns
        schema['inventory_columns'] = self._detect_inventory_columns(df)

        # Calculate inventory metrics
        schema['inventory_metrics'] = self._calculate_inventory_metrics(df, schema)

        # Suggest metrics
        schema['suggested_metrics'] = self._suggest_metrics(schema)

        # Find potential primary keys
        schema['primary_keys'] = self._find_primary_keys(schema)

        return schema

    def get_schema_prompt(self, df: pd.DataFrame) -> str:
        """Get a concise schema description for prompts"""
        schema = self.detect_schema(df)
        return json.dumps({
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
        }, indent=2)

    def _analyze_column(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Analyze a single column in detail"""
        series = df[col]
        non_null = series.dropna()

        info = {
            'name': col,
            'data_type': str(series.dtype),
            'null_count': series.isna().sum(),
            'null_percentage': (series.isna().sum() / len(series)) * 100,
            'unique_count': series.nunique(),
            'is_unique': series.nunique() == len(series),
            'is_id': False,
            'sample_values': series.head(5).tolist(),
            'numeric_range': None,
            'date_range': None,
            'categories': [],
            'string_patterns': None,
            'data_distribution': None,
            'is_inventory_related': False
        }

        # Check if column is inventory-related
        col_lower = col.lower()
        for keyword in self.inventory_keywords:
            if keyword in col_lower:
                info['is_inventory_related'] = True
                break

        # Determine actual data type
        if len(non_null) > 0:
            # Check if numeric
            if pd.api.types.is_numeric_dtype(series):
                info['data_type'] = 'numeric'
                numeric_series = pd.to_numeric(series, errors='coerce').dropna()
                if not numeric_series.empty:
                    numeric_series = numeric_series.astype('float64')
                    info['numeric_range'] = {
                        'min': self._safe_float(numeric_series.min()),
                        'max': self._safe_float(numeric_series.max()),
                        'mean': self._safe_float(numeric_series.mean()),
                        'median': self._safe_float(numeric_series.median()),
                        'std': self._safe_float(numeric_series.std())
                    }
                # Check if it could be an ID (numeric ID)
                if info['is_unique'] and not numeric_series.empty and numeric_series.min() >= 0:
                    info['is_id'] = True

            # Check if date
            elif self._is_date_column(series):
                info['data_type'] = 'date'
                date_series = pd.to_datetime(series, errors='coerce', format='mixed', dayfirst=False)
                if not date_series.isna().all():
                    info['date_range'] = {
                        'min': date_series.min().isoformat(),
                        'max': date_series.max().isoformat()
                    }

            # Check if categorical
            elif info['unique_count'] <= 50:
                info['data_type'] = 'categorical'
                info['categories'] = series.value_counts().head(10).to_dict()

            # Check if string
            else:
                info['data_type'] = 'string'
                # Check for common string patterns
                sample_str = str(non_null.iloc[0]) if len(non_null) > 0 else ''
                if re.match(r'^[A-Z0-9\-]+$', sample_str):
                    info['string_patterns'] = 'alphanumeric_with_hyphens'
                elif re.match(r'^[0-9]+$', sample_str):
                    info['string_patterns'] = 'numeric_string'
                elif re.match(r'^[A-Z]+$', sample_str):
                    info['string_patterns'] = 'uppercase_letters'

        # Check if it's likely an ID column
        if self._is_id_column(series):
            info['is_id'] = True

        return info

    @staticmethod
    def _safe_float(value):
        """Convert numeric stats to JSON-safe floats without crashing on NA/NaN."""
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return None

    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a column contains date values"""
        name_lower = str(series.name).lower()
        if any(token in name_lower for token in ['date', '_dt', ' datetime', 'timestamp']):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                converted = pd.to_datetime(series.dropna().head(50), errors='coerce', format='mixed')
            if len(converted) > 0 and converted.notna().mean() >= 0.7:
                return True

        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False

        date_count = 0
        for val in sample:
            if isinstance(val, (pd.Timestamp, datetime)):
                date_count += 1
            elif isinstance(val, str):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    parsed = pd.to_datetime(pd.Series([val]), errors='coerce', format='mixed')
                if parsed.notna().iloc[0]:
                    date_count += 1
                    continue
                for date_format in self.date_formats:
                    try:
                        datetime.strptime(val.strip(), date_format)
                        date_count += 1
                        break
                    except:
                        continue
            elif isinstance(val, (int, float)):
                # Check if it's a timestamp
                if 1000000000 < val < 9999999999:  # Unix timestamp range
                    try:
                        datetime.fromtimestamp(val)
                        date_count += 1
                    except:
                        pass

        return date_count >= len(sample) * 0.7  # At least 70% are dates

    def _is_id_column(self, series: pd.Series) -> bool:
        """Check if a column is likely an ID column"""
        non_null = series.dropna()
        if len(non_null) == 0:
            return False

        # Check if all non-null values are unique (compare against non_null
        # length, not the full series — otherwise NaN rows cause false negatives
        # on ID columns that have even one missing value).
        if series.nunique() != len(non_null):
            return False

        # Check naming patterns
        name_lower = str(series.name).lower()
        id_patterns = ['id', 'key', 'code', 'number', 'num', 'no', 'sk', 'pk', 'fk']
        if any(pattern in name_lower for pattern in id_patterns):
            return True

        # Check value patterns
        sample = str(non_null.iloc[0]) if len(non_null) > 0 else ''
        if re.match(r'^[A-Z0-9]{5,}$', sample):  # Alphanumeric codes
            return True

        return False

    def _detect_dataset_type(self, schema: Dict[str, Any], df: pd.DataFrame) -> str:
        """Detect the type of dataset based on columns and values"""
        column_names = [col.lower() for col in schema['columns'].keys()]
        column_text = ' '.join(column_names)

        # Check for inventory dataset
        inventory_keywords = ['stock', 'inventory', 'warehouse', 'supply', 'reorder',
                              'turnover', 'batch', 'lot', 'shelf', 'bin', 'pallet']
        if any(keyword in column_text for keyword in inventory_keywords):
            # Check additional inventory indicators
            inventory_score = 0
            for keyword in inventory_keywords:
                if keyword in column_text:
                    inventory_score += 1

            if inventory_score >= 2:
                return 'inventory_management'

            # Check for stock quantity columns
            stock_columns = ['stock', 'quantity', 'on_hand', 'inventory']
            if any(col in column_text for col in stock_columns):
                return 'inventory_management'

        # Check for sales/order dataset
        sales_keywords = ['order', 'sales', 'revenue', 'profit', 'product', 'customer']
        if any(keyword in column_text for keyword in sales_keywords):
            return 'sales_orders'

        # Check for transactions dataset
        transaction_keywords = ['transaction', 'payment', 'invoice', 'bill', 'amount']
        if any(keyword in column_text for keyword in transaction_keywords):
            return 'transactions'

        # Check for customer dataset
        customer_keywords = ['customer', 'client', 'user', 'person', 'contact']
        if any(keyword in column_text for keyword in customer_keywords):
            return 'customer_data'

        # Check for product dataset
        product_keywords = ['product', 'item', 'inventory', 'stock', 'supply']
        if any(keyword in column_text for keyword in product_keywords):
            return 'product_data'

        return 'general'

    def _detect_inventory_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect inventory-specific columns"""
        inventory_columns = {
            'stock_level': None,
            'reorder_point': None,
            'reorder_quantity': None,
            'lead_time': None,
            'turnover': None,
            'warehouse': None,
            'batch_number': None,
            'serial_number': None,
            'supplier': None,
            'expiry_date': None,
            'manufacture_date': None,
            'storage_location': None,
            'condition': None,
            'product_name': None
        }

        for col in df.columns:
            col_lower = col.lower()

            # Map inventory columns
            if any(pattern in col_lower for pattern in ['stock_level', 'current_stock', 'on_hand', 'stock_on_hand']):
                inventory_columns['stock_level'] = col
            elif any(pattern in col_lower for pattern in
                     ['reorder_point', 'reorder_level', 'min_stock', 'reorder_threshold']):
                inventory_columns['reorder_point'] = col
            elif any(pattern in col_lower for pattern in ['reorder_quantity', 'reorder_qty', 'order_qty']):
                inventory_columns['reorder_quantity'] = col
            elif any(pattern in col_lower for pattern in ['lead_time', 'delivery_time', 'supply_lead_time']):
                inventory_columns['lead_time'] = col
            elif any(pattern in col_lower for pattern in ['turnover', 'inventory_turnover', 'stock_turnover']):
                inventory_columns['turnover'] = col
            elif any(pattern in col_lower for pattern in ['warehouse', 'facility', 'storage']):
                inventory_columns['warehouse'] = col
            elif any(pattern in col_lower for pattern in ['batch', 'lot', 'batch_no']):
                inventory_columns['batch_number'] = col
            elif any(pattern in col_lower for pattern in ['serial', 's/n', 'serial_no']):
                inventory_columns['serial_number'] = col
            elif any(pattern in col_lower for pattern in ['supplier', 'vendor', 'manufacturer']):
                inventory_columns['supplier'] = col
            elif any(pattern in col_lower for pattern in ['expiry', 'expiration', 'exp_date']):
                inventory_columns['expiry_date'] = col
            elif any(pattern in col_lower for pattern in ['manufacture', 'mfg', 'production_date']):
                inventory_columns['manufacture_date'] = col
            elif any(pattern in col_lower for pattern in ['storage', 'shelf', 'bin', 'location']):
                inventory_columns['storage_location'] = col
            elif any(pattern in col_lower for pattern in ['condition', 'status', 'state']):
                inventory_columns['condition'] = col
            elif any(pattern in col_lower for pattern in ['product_name', 'item_name', 'product_description']):
                inventory_columns['product_name'] = col

        return inventory_columns

    def _calculate_inventory_metrics(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate inventory-specific metrics"""
        metrics = {}
        inventory_cols = schema.get('inventory_columns', {})

        # Check if we have stock level data
        stock_col = inventory_cols.get('stock_level')
        if stock_col and stock_col in df.columns:
            try:
                stock_data = df[stock_col].dropna()
                if len(stock_data) > 0:
                    metrics['total_stock'] = float(stock_data.sum())
                    metrics['avg_stock'] = float(stock_data.mean())
                    metrics['max_stock'] = float(stock_data.max())
                    metrics['min_stock'] = float(stock_data.min())
                    metrics['zero_stock_count'] = int((stock_data == 0).sum())
            except:
                pass

            # Calculate stock value if cost column exists
            cost_col = None
            for col in df.columns:
                if 'cost' in col.lower() or 'price' in col.lower():
                    cost_col = col
                    break

            if cost_col and cost_col != stock_col:
                try:
                    value_data = df[stock_col] * df[cost_col]
                    metrics['total_inventory_value'] = float(value_data.sum())
                    metrics['avg_inventory_value'] = float(value_data.mean())
                except:
                    pass

        # Check reorder point
        reorder_col = inventory_cols.get('reorder_point')
        if reorder_col and reorder_col in df.columns and stock_col:
            try:
                low_stock = df[stock_col] <= df[reorder_col]
                metrics['low_stock_count'] = int(low_stock.sum())
                metrics['low_stock_percentage'] = float((low_stock.sum() / len(df)) * 100)
            except:
                pass

        return metrics

    def _map_columns(self, schema: Dict[str, Any]) -> Dict[str, str]:
        """Map columns to standardized names.

        Two-pass strategy:
          1. Exact match pass - a column whose (lowercased) name exactly
             equals the standard name or one of its listed synonyms always
             wins, for every standard name.
          2. Substring fallback pass - only for standard names that still
             have no mapping after pass 1, take the *first* column whose
             name contains one of the synonyms as a substring.

        The previous implementation looped over columns and used `break`
        inside the substring-matching inner loop, but that break only
        exited the inner `for variation in variations` loop - not the outer
        `for col in columns` loop - so it kept scanning every remaining
        column and could silently overwrite a correct match with an
        unrelated, worse, later match.
        """
        mapping = {}
        columns = list(schema['columns'].keys())

        # Pass 1: exact matches (highest priority)
        for standard_name, variations in self.column_mappings.items():
            for col in columns:
                col_lower = col.lower()
                if col_lower == standard_name or col_lower in variations:
                    mapping[standard_name] = col
                    break

        # Pass 2: substring fallback, only for still-unmapped standard names.
        # Uses word-boundary matching on an underscore/hyphen-normalized
        # form of both the column name and the synonym, so a short synonym
        # like 'count' (for 'quantity') doesn't falsely match inside an
        # unrelated column name such as 'Country' or 'Discount', while
        # compound names like 'sub_category' still correctly match the
        # 'category' synonym (a bare regex \b wouldn't see a boundary at
        # the underscore, since '_' counts as a word character).
        def _normalize(s):
            return re.sub(r'[_\-]+', ' ', s.lower()).strip()

        for standard_name, variations in self.column_mappings.items():
            if standard_name in mapping:
                continue
            for col in columns:
                col_norm = _normalize(col)
                if any(re.search(rf'\b{re.escape(_normalize(v))}\b', col_norm) for v in variations):
                    mapping[standard_name] = col
                    break

        return mapping

    def _suggest_metrics(self, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest metrics based on schema"""
        metrics = []
        mapping = schema['column_mapping']
        dataset_type = schema.get('dataset_type', 'general')
        inventory_cols = schema.get('inventory_columns', {})

        # Inventory metrics
        if dataset_type == 'inventory_management':
            # Stock metrics
            if inventory_cols.get('stock_level'):
                metrics.append({
                    'name': 'Total Stock Quantity',
                    'description': 'Sum of all stock levels',
                    'query': f"SELECT SUM({inventory_cols['stock_level']}) as Total_Stock FROM {{table}}",
                    'type': 'numeric',
                    'category': 'inventory'
                })

                metrics.append({
                    'name': 'Average Stock Level',
                    'description': 'Average stock level across all products',
                    'query': f"SELECT AVG({inventory_cols['stock_level']}) as Avg_Stock FROM {{table}}",
                    'type': 'numeric',
                    'category': 'inventory'
                })

            # Reorder metrics
            if inventory_cols.get('reorder_point') and inventory_cols.get('stock_level'):
                metrics.append({
                    'name': 'Low Stock Items',
                    'description': 'Number of items below reorder point',
                    'query': f"""
                        SELECT COUNT(*) as Low_Stock_Items 
                        FROM {{table}} 
                        WHERE {inventory_cols['stock_level']} <= {inventory_cols['reorder_point']}
                    """,
                    'type': 'count',
                    'category': 'inventory'
                })

                metrics.append({
                    'name': 'Out of Stock Items',
                    'description': 'Number of items with zero stock',
                    'query': f"""
                        SELECT COUNT(*) as Out_Of_Stock 
                        FROM {{table}} 
                        WHERE {inventory_cols['stock_level']} = 0
                    """,
                    'type': 'count',
                    'category': 'inventory'
                })

            # Turnover metrics
            if inventory_cols.get('turnover'):
                metrics.append({
                    'name': 'Average Turnover Rate',
                    'description': 'Average inventory turnover rate',
                    'query': f"SELECT AVG({inventory_cols['turnover']}) as Avg_Turnover FROM {{table}}",
                    'type': 'numeric',
                    'category': 'inventory'
                })

            # Supplier metrics
            if inventory_cols.get('supplier'):
                metrics.append({
                    'name': 'Top Suppliers by Stock',
                    'description': 'Suppliers with highest total stock',
                    'query': f"""
                        SELECT {inventory_cols['supplier']} as Supplier,
                               SUM({inventory_cols['stock_level']}) as Total_Stock
                        FROM {{table}} 
                        GROUP BY {inventory_cols['supplier']}
                        ORDER BY Total_Stock DESC
                        LIMIT 5
                    """,
                    'type': 'list',
                    'category': 'inventory'
                })

            # Warehouse metrics
            if inventory_cols.get('warehouse'):
                metrics.append({
                    'name': 'Stock by Warehouse',
                    'description': 'Stock distribution across warehouses',
                    'query': f"""
                        SELECT {inventory_cols['warehouse']} as Warehouse,
                               SUM({inventory_cols['stock_level']}) as Total_Stock,
                               COUNT(*) as Product_Count
                        FROM {{table}} 
                        GROUP BY {inventory_cols['warehouse']}
                        ORDER BY Total_Stock DESC
                    """,
                    'type': 'list',
                    'category': 'inventory'
                })

            # Expiry metrics
            if inventory_cols.get('expiry_date'):
                metrics.append({
                    'name': 'Expiring Soon',
                    'description': 'Items expiring within 30 days',
                    'query': f"""
                        SELECT {inventory_cols.get('product_name', 'Product_Name')} as Product,
                               {inventory_cols['expiry_date']} as Expiry_Date
                        FROM {{table}}
                        WHERE {inventory_cols['expiry_date']} BETWEEN DATE('now') AND DATE('now', '+30 days')
                        ORDER BY {inventory_cols['expiry_date']} ASC
                    """,
                    'type': 'list',
                    'category': 'inventory'
                })

        # Sales metrics
        if 'sales' in mapping:
            metrics.append({
                'name': 'Total Revenue',
                'description': 'Sum of all sales/revenue',
                'query': f"SELECT ROUND(SUM({mapping['sales']}), 2) as Total_Revenue FROM {{table}}",
                'type': 'numeric',
                'category': 'sales'
            })

        if 'profit' in mapping:
            metrics.append({
                'name': 'Total Profit',
                'description': 'Sum of all profits',
                'query': f"SELECT ROUND(SUM({mapping['profit']}), 2) as Total_Profit FROM {{table}}",
                'type': 'numeric',
                'category': 'sales'
            })

        if 'order_id' in mapping:
            metrics.append({
                'name': 'Total Orders',
                'description': 'Number of unique orders',
                'query': f"SELECT COUNT(DISTINCT {mapping['order_id']}) as Total_Orders FROM {{table}}",
                'type': 'count',
                'category': 'sales'
            })

        # Customer metrics
        if 'customer_id' in mapping:
            metrics.append({
                'name': 'Total Customers',
                'description': 'Number of unique customers',
                'query': f"SELECT COUNT(DISTINCT {mapping['customer_id']}) as Total_Customers FROM {{table}}",
                'type': 'count',
                'category': 'customer'
            })

        if 'customer_id' in mapping and 'customer_name' in mapping and 'sales' in mapping:
            profit_select = ""
            if 'profit' in mapping:
                profit_select = f", ROUND(SUM({mapping['profit']}), 2) as Total_Profit"
            metrics.append({
                'name': 'Top Customers by Revenue',
                'description': 'Customer-level revenue grouped by customer id and name',
                'query': f"""
                    SELECT {mapping['customer_id']} as Customer_ID,
                           {mapping['customer_name']} as Customer_Name,
                           ROUND(SUM({mapping['sales']}), 2) as Total_Revenue
                           {profit_select},
                           COUNT(DISTINCT {mapping.get('order_id', mapping['customer_id'])}) as Order_Count
                    FROM {{table}}
                    GROUP BY {mapping['customer_id']}, {mapping['customer_name']}
                    ORDER BY Total_Revenue DESC
                    LIMIT 10
                """,
                'type': 'list',
                'category': 'customer'
            })

        # Product metrics
        if 'product_id' in mapping:
            metrics.append({
                'name': 'Total Products',
                'description': 'Number of unique products',
                'query': f"SELECT COUNT(DISTINCT {mapping['product_id']}) as Total_Products FROM {{table}}",
                'type': 'count',
                'category': 'product'
            })

        date_col = mapping.get('order_date') or mapping.get('transaction_date') or mapping.get('date')

        # Date range metrics
        if date_col:
            metrics.append({
                'name': 'Date Range',
                'description': 'Date range of data',
                'query': f"SELECT MIN({date_col}) as Start_Date, MAX({date_col}) as End_Date FROM {{table}}",
                'type': 'date_range',
                'category': 'general'
            })

        if date_col and 'sales' in mapping:
            _iso = f"{date_col} LIKE '____-__-__%'"
            _year = (f"CASE WHEN {_iso} THEN strftime('%Y', {date_col}) "
                     f"ELSE substr({date_col}, -4) END")
            _month = (f"CASE WHEN {_iso} THEN strftime('%Y-%m', {date_col}) "
                      f"ELSE substr({date_col}, -4) || '-' || "
                      f"printf('%02d', CAST(substr({date_col}, 1, instr({date_col}, '/') - 1) AS INTEGER)) END")
            _filter = f"{date_col} IS NOT NULL"

            metrics.append({
                'name': 'Monthly Revenue Trend',
                'description': 'Revenue by order month',
                'query': f"""
                    SELECT {_month} as Month,
                           ROUND(SUM({mapping['sales']}), 2) as Revenue
                    FROM {{table}}
                    WHERE {_filter}
                    GROUP BY {_month}
                    ORDER BY Month
                """,
                'type': 'time_series',
                'category': 'time_intelligence'
            })

            metrics.append({
                'name': 'YoY Revenue',
                'description': 'Revenue by year with year-over-year change',
                'query': f"""
                    WITH yearly AS (
                        SELECT {_year} as Year,
                               ROUND(SUM({mapping['sales']}), 2) as Revenue
                        FROM {{table}}
                        WHERE {_filter}
                        GROUP BY {_year}
                    )
                    SELECT Year,
                           Revenue,
                           ROUND(Revenue - LAG(Revenue) OVER (ORDER BY Year), 2) as YoY_Revenue_Change,
                           ROUND(
                               (Revenue - LAG(Revenue) OVER (ORDER BY Year))
                               / NULLIF(LAG(Revenue) OVER (ORDER BY Year), 0) * 100,
                               2
                           ) as YoY_Revenue_Change_Pct
                    FROM yearly
                    ORDER BY Year
                """,
                'type': 'time_series',
                'category': 'time_intelligence'
            })

        if date_col and 'profit' in mapping:
            _iso = f"{date_col} LIKE '____-__-__%'"
            _month = (f"CASE WHEN {_iso} THEN strftime('%Y-%m', {date_col}) "
                      f"ELSE substr({date_col}, -4) || '-' || "
                      f"printf('%02d', CAST(substr({date_col}, 1, instr({date_col}, '/') - 1) AS INTEGER)) END")
            _filter = f"{date_col} IS NOT NULL"

            metrics.append({
                'name': 'Monthly Profit Trend',
                'description': 'Profit by order month',
                'query': f"""
                    SELECT {_month} as Month,
                           ROUND(SUM({mapping['profit']}), 2) as Profit
                    FROM {{table}}
                    WHERE {_filter}
                    GROUP BY {_month}
                    ORDER BY Month
                """,
                'type': 'time_series',
                'category': 'time_intelligence'
            })

        return metrics

    def _find_primary_keys(self, schema: Dict[str, Any]) -> List[str]:
        """Find potential primary keys"""
        potential_keys = []

        # Check ID columns first
        for col in schema.get('id_columns', []):
            if schema['columns'][col].get('is_unique', False):
                potential_keys.append(col)

        # Check unique columns with low null count
        for col in schema.get('unique_columns', []):
            if col not in potential_keys:
                null_pct = schema['columns'][col].get('null_percentage', 100)
                if null_pct < 5:
                    potential_keys.append(col)

        # If no unique keys found, use first column
        if not potential_keys and schema['columns']:
            potential_keys.append(list(schema['columns'].keys())[0])

        return potential_keys