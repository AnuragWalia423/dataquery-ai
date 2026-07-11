# data_normalizer.py
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import re
from config import Config


class DataNormalizer:
    def __init__(self):
        self.column_mappings = Config.COLUMN_MAPPINGS
        self.date_formats = Config.DATE_FORMATS
        self.cleaning_rules = {}
        self.last_report = {}

    def normalize_dataframe(self, df: pd.DataFrame, schema: Dict[str, Any], level: str = 'advanced') -> pd.DataFrame:
        """Normalize a DataFrame based on detected schema.

        level controls how much processing runs, matching the three
        options offered in the app.py "Cleaning Level" selector:
          - 'basic':    just remove duplicates / handle missing values
                        (plus the always-needed structural steps: column
                        name standardization and numeric type coercion)
          - 'standard': basic + normalize dates and strings
          - 'advanced': standard + categorical standardization, metadata
                        columns, and inventory-specific normalization

        Previously this parameter didn't exist at all - the UI offered a
        Basic/Standard/Advanced choice but the selection was never read,
        so every level silently ran the full advanced pipeline.
        """
        self._reset_report(df)
        df_copy = df.copy()
        dataset_type = schema.get('dataset_type', 'general')
        level = (level or 'advanced').lower()
        if level not in ('basic', 'standard', 'advanced'):
            level = 'advanced'
        self.last_report['cleaning_level'] = level
        self.last_report['dataset_type'] = dataset_type or 'general'

        # Always run: structural normalization needed for every downstream
        # step to work reliably against predictable column names/types.
        #
        # schema was computed on the ORIGINAL (pre-rename) dataframe, so it
        # still refers to columns by their original names (e.g.
        # 'Customer_ID'). Once we rename columns here, every schema field
        # that holds a column name (column_mapping, inventory_columns,
        # id_columns, primary_keys, date/numeric/string_columns) must be
        # remapped to the new names too, or every later lookup against
        # schema silently (or not-so-silently) breaks.
        df_copy = self._drop_empty_rows_and_columns(df_copy)
        df_copy, rename_map = self._standardize_column_names(df_copy)
        schema = self._remap_schema_columns(schema, rename_map)

        df_copy = self._standardize_missing_tokens(df_copy)
        df_copy = self._coerce_object_columns(df_copy)
        df_copy = self._normalize_numerics(df_copy, schema)
        df_copy = self._handle_missing_values(df_copy, schema)
        df_copy = self._remove_duplicates(df_copy, schema)

        if level in ('standard', 'advanced'):
            df_copy = self._clean_string_columns(df_copy)
            df_copy = self._normalize_dates(df_copy, schema)

        if level == 'advanced':
            df_copy = self._clean_categorical_columns(df_copy, schema)
            df_copy = self._add_metadata_columns(df_copy)
            if dataset_type == 'inventory_management':
                df_copy = self._normalize_inventory_data(df_copy, schema)

        self.last_report['final_rows'] = int(len(df_copy))
        self.last_report['final_columns'] = int(len(df_copy.columns))
        self.last_report['rows_removed'] = int(self.last_report['original_rows'] - len(df_copy))
        self.last_report['columns_added'] = int(len(df_copy.columns) - self.last_report['original_columns'])
        return df_copy

    def get_last_report(self) -> Dict[str, Any]:
        """Return a JSON-safe summary of the most recent cleaning run."""
        return dict(self.last_report or {})

    def _reset_report(self, df: pd.DataFrame) -> None:
        self.last_report = {
            'original_rows': int(len(df)),
            'original_columns': int(len(df.columns)),
            'final_rows': int(len(df)),
            'final_columns': int(len(df.columns)),
            'rows_removed': 0,
            'columns_added': 0,
            'empty_rows_removed': 0,
            'empty_columns_removed': [],
            'renamed_columns': {},
            'deduplicated_columns': {},
            'missing_tokens_replaced': 0,
            'type_conversions': {},
            'missing_values_filled': {},
            'duplicates_removed': 0,
            'date_columns_parsed': [],
            'derived_date_columns_added': [],
            'metadata_columns_added': [],
            'inventory_columns_added': [],
        }

    @staticmethod
    def _remap_schema_columns(schema: Dict[str, Any], rename_map: Dict[str, str]) -> Dict[str, Any]:
        """Return a copy of schema with every column-name reference updated
        to match the post-rename column names, using rename_map (old -> new).
        Names not present in rename_map are left as-is.
        """
        def remap_name(name):
            return rename_map.get(name, name) if name else name

        remapped = dict(schema)

        if remapped.get('column_mapping'):
            remapped['column_mapping'] = {
                std_name: remap_name(col) for std_name, col in remapped['column_mapping'].items()
            }

        if remapped.get('inventory_columns'):
            remapped['inventory_columns'] = {
                col_type: remap_name(col) for col_type, col in remapped['inventory_columns'].items()
            }

        for list_field in ('id_columns', 'primary_keys', 'date_columns',
                           'numeric_columns', 'string_columns', 'unique_columns'):
            if remapped.get(list_field):
                remapped[list_field] = [remap_name(c) for c in remapped[list_field]]

        return remapped

    def _drop_empty_rows_and_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows/columns that contain no real data at all."""
        before_rows = len(df)
        before_cols = list(df.columns)

        df = df.dropna(axis=0, how='all')
        df = df.dropna(axis=1, how='all')

        self.last_report['empty_rows_removed'] = int(before_rows - len(df))
        removed_cols = [str(col) for col in before_cols if col not in df.columns]
        self.last_report['empty_columns_removed'] = removed_cols
        return df

    def _normalize_inventory_data(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Apply inventory-specific normalization"""
        inventory_cols = schema.get('inventory_columns', {})
        df_copy = df.copy()

        # Normalize stock levels
        if inventory_cols.get('stock_level'):
            stock_col = inventory_cols['stock_level']
            if stock_col in df_copy.columns:
                # Ensure stock levels are non-negative
                df_copy[stock_col] = df_copy[stock_col].clip(lower=0)

                # Create stock status categories
                reorder_col = inventory_cols.get('reorder_point')
                if reorder_col and reorder_col in df_copy.columns:
                    df_copy['stock_status'] = 'In Stock'
                    df_copy.loc[df_copy[stock_col] == 0, 'stock_status'] = 'Out of Stock'
                    df_copy.loc[df_copy[stock_col] <= df_copy[reorder_col], 'stock_status'] = 'Low Stock'
                    df_copy.loc[df_copy[stock_col] > df_copy[reorder_col] * 3, 'stock_status'] = 'Overstock'

        # Calculate inventory value
        cost_col = None
        for col in df_copy.columns:
            if 'cost' in col.lower() or 'unit_cost' in col.lower():
                cost_col = col
                break

        if cost_col and inventory_cols.get('stock_level'):
            stock_col = inventory_cols['stock_level']
            if stock_col in df_copy.columns and cost_col in df_copy.columns:
                if 'inventory_value' not in df_copy.columns:
                    df_copy['inventory_value'] = df_copy[stock_col] * df_copy[cost_col]
                    self.last_report['inventory_columns_added'].append('inventory_value')

        # Calculate days to expiry
        if inventory_cols.get('expiry_date'):
            expiry_col = inventory_cols['expiry_date']
            if expiry_col in df_copy.columns and pd.api.types.is_datetime64_dtype(df_copy[expiry_col]):
                if 'days_to_expiry' not in df_copy.columns:
                    df_copy['days_to_expiry'] = (df_copy[expiry_col] - pd.Timestamp.now()).dt.days
                    self.last_report['inventory_columns_added'].append('days_to_expiry')

        # Normalize warehouse names
        if inventory_cols.get('warehouse'):
            warehouse_col = inventory_cols['warehouse']
            if warehouse_col in df_copy.columns:
                # Standardize warehouse names
                df_copy[warehouse_col] = df_copy[warehouse_col].str.upper().str.strip()
                df_copy[warehouse_col] = df_copy[warehouse_col].str.replace(r'\s+', ' ', regex=True)

        # Normalize batch numbers
        if inventory_cols.get('batch_number'):
            batch_col = inventory_cols['batch_number']
            if batch_col in df_copy.columns:
                # Standardize batch number format
                df_copy[batch_col] = df_copy[batch_col].str.strip().str.upper()
                df_copy[batch_col] = df_copy[batch_col].str.replace(r'[^A-Z0-9\-]', '', regex=True)

        # Normalize supplier names
        if inventory_cols.get('supplier'):
            supplier_col = inventory_cols['supplier']
            if supplier_col in df_copy.columns:
                df_copy[supplier_col] = df_copy[supplier_col].str.strip().str.upper()

        return df_copy

    def _standardize_column_names(self, df: pd.DataFrame):
        """Standardize column names.

        Returns (renamed_df, rename_map) where rename_map maps each
        original column name to its new standardized name - needed so
        normalize_dataframe can keep the schema's column-name references
        in sync with the renamed dataframe.
        """
        raw_columns = []

        for col in df.columns:
            # Convert to lowercase
            clean_col = str(col).lower().strip()

            # Remove special characters
            clean_col = re.sub(r'[^a-z0-9_]', '_', clean_col)

            # Remove multiple underscores
            clean_col = re.sub(r'_+', '_', clean_col)

            # Remove trailing underscores
            clean_col = clean_col.strip('_')

            # Handle common inventory-specific column names
            if 'stock' in clean_col and 'level' in clean_col:
                clean_col = 'stock_level'
            elif 'reorder' in clean_col and 'point' in clean_col:
                clean_col = 'reorder_point'
            elif 'reorder' in clean_col and 'qty' in clean_col:
                clean_col = 'reorder_quantity'
            elif 'lead' in clean_col and 'time' in clean_col:
                clean_col = 'lead_time'
            elif 'turnover' in clean_col:
                clean_col = 'turnover'
            elif 'warehouse' in clean_col or 'facility' in clean_col:
                clean_col = 'warehouse'
            elif 'batch' in clean_col or 'lot' in clean_col:
                clean_col = 'batch_number'
            elif 'serial' in clean_col:
                clean_col = 'serial_number'
            elif 'supplier' in clean_col or 'vendor' in clean_col:
                clean_col = 'supplier'
            elif 'expiry' in clean_col or 'expiration' in clean_col:
                clean_col = 'expiry_date'
            elif 'manufacture' in clean_col or 'mfg' in clean_col:
                clean_col = 'manufacture_date'

            if not clean_col:
                clean_col = 'column'

            raw_columns.append(clean_col)

        unique_columns = self._make_unique_column_names(raw_columns)
        rename_map = dict(zip(df.columns, unique_columns))
        deduplicated = {
            raw: unique
            for raw, unique in zip(raw_columns, unique_columns)
            if raw != unique
        }

        self.last_report['renamed_columns'] = {
            str(old): new for old, new in rename_map.items() if str(old) != new
        }
        self.last_report['deduplicated_columns'] = deduplicated

        return df.rename(columns=rename_map), rename_map

    @staticmethod
    def _make_unique_column_names(columns):
        seen = {}
        unique = []
        for col in columns:
            base = col or 'column'
            count = seen.get(base, 0)
            if count == 0:
                unique_col = base
            else:
                unique_col = f"{base}_{count + 1}"
            while unique_col in seen:
                count += 1
                unique_col = f"{base}_{count + 1}"
            seen[base] = count + 1
            seen[unique_col] = 1
            unique.append(unique_col)
        return unique

    def _standardize_missing_tokens(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert common placeholder strings to real missing values."""
        missing_tokens = {
            '', ' ', 'na', 'n/a', 'null', 'none', 'nan', 'nat', '-', '--',
            'unknown', 'not available', 'not_available', '?'
        }
        replaced = 0
        for col in df.select_dtypes(include=['object', 'string']).columns:
            stripped = df[col].astype('string').str.strip()
            mask = stripped.str.lower().isin(missing_tokens)
            replaced += int(mask.sum())
            df.loc[mask, col] = pd.NA
        self.last_report['missing_tokens_replaced'] = replaced
        return df

    def _coerce_object_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Infer safe numeric/boolean types from text columns without hardcoded dataset names."""
        for col in df.select_dtypes(include=['object', 'string']).columns:
            series = df[col]
            non_null = series.dropna()
            if non_null.empty:
                continue

            text = series.astype('string').str.strip()
            lowered = text.str.lower()
            non_null_count = int(text.notna().sum())

            boolean_values = {
                'true': True, 'false': False,
                'yes': True, 'no': False,
                'y': True, 'n': False,
            }
            observed = set(lowered.dropna().unique())
            if observed and observed.issubset(boolean_values.keys()):
                df[col] = lowered.map(boolean_values).astype('boolean')
                self.last_report['type_conversions'][col] = 'boolean'
                continue

            cleaned_numeric = (
                text
                .str.replace(r'^\((.*)\)$', r'-\1', regex=True)
                .str.replace(r'[$€£¥₹,]', '', regex=True)
                .str.replace('%', '', regex=False)
            )
            numeric = pd.to_numeric(cleaned_numeric, errors='coerce')
            numeric_ratio = float(numeric.notna().sum() / non_null_count) if non_null_count else 0

            # Keep ID/code-like text intact even if it contains digits.
            col_lower = col.lower()
            looks_like_identifier = any(token in col_lower for token in ['id', 'code', 'sku', 'zip', 'postal', 'phone', 'serial'])
            looks_like_date = any(token in col_lower for token in ['date', '_dt', 'datetime', 'timestamp'])
            if looks_like_date:
                continue
            if numeric_ratio >= 0.9 and not looks_like_identifier:
                if text.str.contains('%', na=False).any() and numeric.max(skipna=True) > 1:
                    numeric = numeric / 100
                df[col] = numeric
                self.last_report['type_conversions'][col] = 'numeric'

        return df

    def _clean_string_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean string columns.

        IMPORTANT: every rule below used to run on *every* object column
        in the dataframe, regardless of what that column actually held.
        In particular the "standardize phone numbers" rule
        (str.replace(r'[^0-9+ ]', '', regex=True)) strips out every
        letter, so applying it to all string columns would have silently
        blanked out names, addresses, categories, product names - any
        non-numeric text in the whole dataset. The abbreviation-expansion
        rules were also missing `regex=True`, which on pandas>=2.0 means
        they were silent no-ops (regex defaults to False there) - so
        fixing that without also scoping the rules to the right columns
        would have turned a "does nothing" bug into a "corrupts data"
        bug. Each rule is now only applied to columns whose name suggests
        it's actually relevant.
        """
        phone_pattern = re.compile(r'phone|mobile|contact_no|fax')
        address_pattern = re.compile(r'address|street|addr')
        unit_pattern = re.compile(r'unit|uom|measure|pack')
        condition_pattern = re.compile(r'condition|status|state\b')
        money_pattern = re.compile(r'price|cost|amount|sales|revenue|profit|fee|value')

        for col in df.select_dtypes(include=['object', 'string']).columns:
            col_lower = col.lower()

            # Convert to string and do the always-safe generic cleanup
            df[col] = df[col].astype(str)
            df[col] = df[col].str.strip()
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)

            # Currency symbols only matter on columns that hold monetary
            # text (most monetary columns are numeric dtype already, but
            # some datasets keep "$1,234.56" as text)
            if money_pattern.search(col_lower):
                df[col] = df[col].str.replace(r'[$€£¥]', '', regex=True)

            # Phone-number style stripping - only for phone/contact columns
            if phone_pattern.search(col_lower):
                df[col] = df[col].str.replace(r'[^0-9+ ]', '', regex=True)
                continue  # nothing else below is relevant to a phone column

            # Address abbreviation expansion - only for address columns
            if address_pattern.search(col_lower):
                df[col] = df[col].str.replace(r'(?i)\bst\b', 'Street', regex=True)
                df[col] = df[col].str.replace(r'(?i)\bav\b', 'Avenue', regex=True)
                df[col] = df[col].str.replace(r'(?i)\bdr\b', 'Drive', regex=True)
                df[col] = df[col].str.replace(r'(?i)\brd\b', 'Road', regex=True)
                df[col] = df[col].str.replace(r'(?i)\bpl\b', 'Place', regex=True)

            # Unit-of-measure standardization - only for unit/quantity-label columns
            if unit_pattern.search(col_lower):
                df[col] = df[col].str.replace(r'(?i)\bpcs?\b', 'units', regex=True)
                df[col] = df[col].str.replace(r'(?i)\blbs?\b', 'lbs', regex=True)
                df[col] = df[col].str.replace(r'(?i)\bctn\b', 'carton', regex=True)

            # Condition/status word standardization - only for condition/status columns
            if condition_pattern.search(col_lower):
                df[col] = df[col].str.replace(r'(?i)\bnew\b', 'New', regex=True)
                df[col] = df[col].str.replace(r'(?i)\bused\b', 'Used', regex=True)
                df[col] = df[col].str.replace(r'(?i)\brefurbished\b', 'Refurbished', regex=True)
                df[col] = df[col].str.replace(r'(?i)\bdamaged\b', 'Damaged', regex=True)

        return df

    def _normalize_dates(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Normalize date columns"""
        mapped_columns = schema.get('column_mapping', {})
        inventory_cols = schema.get('inventory_columns', {})

        # Get all potential date columns
        potential_dates = []
        for col in df.columns:
            col_lower = col.lower()
            if any(date_keyword in col_lower for date_keyword in ['date', 'time', 'day', 'month', 'year']):
                potential_dates.append(col)

        # Add mapped date columns
        date_types = ['order_date', 'ship_date', 'invoice_date', 'payment_date',
                      'stock_date', 'expiry_date', 'manufacture_date', 'restock_date']
        for date_type in date_types:
            if date_type in mapped_columns:
                col = mapped_columns[date_type]
                if col in df.columns and col not in potential_dates:
                    potential_dates.append(col)

        # Add inventory date columns
        for inv_type in ['expiry_date', 'manufacture_date', 'restock_date']:
            if inv_type in inventory_cols:
                col = inventory_cols[inv_type]
                if col in df.columns and col not in potential_dates:
                    potential_dates.append(col)

        for col in potential_dates:
            try:
                # Try to convert to datetime. Superstore-style CSVs often
                # contain mixed M/D/YYYY and MM/DD/YYYY strings; pandas'
                # default strict parser can turn valid dates into NaT.
                parsed = self._parse_mixed_date_series(df[col])
                if parsed.notna().sum() == 0:
                    continue
                df[col] = parsed
                if col not in self.last_report['date_columns_parsed']:
                    self.last_report['date_columns_parsed'].append(col)

                # Create derived columns
                derived_columns = {
                    f'{col}_year': df[col].dt.year,
                    f'{col}_month': df[col].dt.month,
                    f'{col}_day': df[col].dt.day,
                    f'{col}_quarter': df[col].dt.quarter,
                    f'{col}_weekday': df[col].dt.weekday,
                    f'{col}_month_name': df[col].dt.month_name(),
                    f'{col}_week': df[col].dt.isocalendar().week,
                }
                for derived_col, values in derived_columns.items():
                    if derived_col not in df.columns:
                        df[derived_col] = values
                        self.last_report['derived_date_columns_added'].append(derived_col)

                # Calculate days difference for inventory
                if 'expiry' in col.lower() or 'expiration' in col.lower():
                    derived_col = f'{col}_days_until'
                    if derived_col not in df.columns:
                        df[derived_col] = (df[col] - pd.Timestamp.now()).dt.days
                        self.last_report['derived_date_columns_added'].append(derived_col)

            except Exception as e:
                print(f"Could not convert {col} to datetime: {e}")

        return df

    @staticmethod
    def _parse_mixed_date_series(series: pd.Series) -> pd.Series:
        """Parse common mixed CSV date formats without turning valid dates into NaT.

        Mirrors a staging-table CASE strategy:
          - YYYY-MM-DD
          - MM-DD-YYYY
          - M/D/YYYY or MM/DD/YYYY
          - timestamp strings and pandas/dateutil fallback
        """
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.to_datetime(series, errors='coerce')

        text = series.astype("string").str.strip()
        parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

        format_rules = [
            (text.str.match(r'^\d{4}-\d{1,2}-\d{1,2}$', na=False), '%Y-%m-%d'),
            (text.str.match(r'^\d{1,2}-\d{1,2}-\d{4}$', na=False), '%m-%d-%Y'),
            (text.str.match(r'^\d{1,2}/\d{1,2}/\d{4}$', na=False), '%m/%d/%Y'),
            (text.str.match(r'^\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}:\d{2}', na=False), None),
        ]

        for mask, date_format in format_rules:
            if not mask.any():
                continue
            if date_format:
                parsed.loc[mask] = pd.to_datetime(text.loc[mask], errors='coerce', format=date_format)
            else:
                parsed.loc[mask] = pd.to_datetime(text.loc[mask], errors='coerce', format='mixed')

        remaining = parsed.isna() & text.notna() & (text != "")
        if remaining.any():
            parsed.loc[remaining] = pd.to_datetime(text.loc[remaining], errors='coerce', format='mixed')

        return parsed

    def _normalize_numerics(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Normalize numeric columns"""
        for col in df.select_dtypes(include=['number']).columns:
            col_lower = col.lower()
            # Check for monetary values
            if any(keyword in col_lower for keyword in ['amount', 'sales', 'profit', 'price', 'cost']):
                # Round to 2 decimal places
                df[col] = df[col].round(2)

            # Check for percentage values
            # Do not treat turnover_rate as a percentage: inventory turnover is
            # a ratio used directly by the turnover analysis buckets.
            if any(keyword in col_lower for keyword in ['percent', 'rate', 'discount']) and 'turnover' not in col_lower:
                # Ensure percentage is between 0 and 1
                if df[col].max() > 1:
                    df[col] = df[col] / 100

            # Check for stock values
            # Check for stock values
            if any(keyword in col_lower for keyword in ['stock', 'quantity', 'qty']):
                # Ensure non-negative integers
                df[col] = (
                    df[col]
                    .fillna(0)
                    .clip(lower=0)
                    .round(0)
                    .astype("Int64")
                )

            # Check for turnover
            if 'turnover' in col_lower:
                # Round to 2 decimal places
                df[col] = df[col].round(2)

        return df

    def _clean_categorical_columns(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Clean categorical columns.

        Each named domain below (yes/no, male/female, new/used/...,
        in stock/out of stock/backorder) is only applied to a column if
        EVERY distinct value in that column already belongs to that
        domain. Previously the mapping was applied unconditionally to
        every categorical column, which meant a short single-letter
        synonym like 'm' (intended for "male") could silently rewrite an
        unrelated column - e.g. a product 'Size' column containing
        S/M/L would have its 'M' rows turned into 'male'.
        """
        categorical_columns = []

        # Find categorical columns
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() <= 50:
                categorical_columns.append(col)

        domains = {
            'yes': ['y', 'yes', 'true', '1', 't'],
            'no': ['n', 'no', 'false', '0', 'f'],
            'male': ['m', 'male', 'man'],
            'female': ['f', 'female', 'woman'],
            'new': ['new', 'nw', 'brand new'],
            'used': ['used', 'second hand', 'pre-owned'],
            'refurbished': ['refurbished', 'reconditioned', 'remanufactured'],
            'damaged': ['damaged', 'defective', 'broken'],
            'in stock': ['in stock', 'available', 'on hand'],
            'out of stock': ['out of stock', 'unavailable', 'sold out'],
            'backorder': ['backorder', 'back order', 'on order']
        }

        # Group domains that represent the same underlying concept, so a
        # column is only normalized if ALL of its distinct values fall
        # within one coherent concept (not just within the flattened union
        # of every domain below - that flattened union is what let 'm'/'f'/
        # '0'/'1' bleed into unrelated columns before).
        concept_groups = [
            {'yes', 'no'},
            {'male', 'female'},
            {'new', 'used', 'refurbished', 'damaged'},
            {'in stock', 'out of stock', 'backorder'},
        ]

        for col in categorical_columns:
            # Convert to string and clean
            df[col] = df[col].astype(str).str.strip()
            col_values = set(df[col].str.lower().unique())
            col_values.discard('')
            col_values.discard('nan')

            if not col_values:
                continue

            for concept in concept_groups:
                allowed_values = set()
                for standard in concept:
                    allowed_values.update(domains[standard])

                if col_values.issubset(allowed_values):
                    df[col] = df[col].str.lower()
                    for standard in concept:
                        mask = df[col].isin(domains[standard])
                        df.loc[mask, col] = standard
                    break

            # Fill empty strings
            df[col] = df[col].replace('', 'unknown')

        return df

    def _handle_missing_values(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Handle missing values based on column type"""
        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            if missing_count == 0:
                continue

            col_lower = col.lower()
            if any(token in col_lower for token in ['date', '_dt', 'datetime', 'timestamp']):
                continue

            # Numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                # For monetary columns, fill with 0
                if any(keyword in col_lower for keyword in ['sales', 'profit', 'amount', 'price', 'cost']):
                    df[col] = df[col].fillna(0)
                # For stock columns, fill with 0
                elif any(keyword in col_lower for keyword in ['stock', 'quantity', 'qty']):
                    df[col] = df[col].fillna(0)
                else:
                    # Fill with median for other numeric columns
                    median = df[col].median()
                    df[col] = df[col].fillna(0 if pd.isna(median) else median)

            # Date columns
            elif pd.api.types.is_datetime64_dtype(df[col]):
                # Do not invent dates. A generic cleaning pipeline should keep
                # unknown dates as null rather than replacing them with today.
                continue

            # Categorical columns
            elif df[col].dtype == 'object' and df[col].nunique() <= 50:
                df[col] = df[col].fillna('unknown')

            # String columns
            else:
                df[col] = df[col].fillna('not_available')

            self.last_report['missing_values_filled'][col] = missing_count

        return df

    def _remove_duplicates(self, df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
        """Remove duplicates based on key columns"""
        before = len(df)
        # Try to use ID columns first
        id_columns = schema.get('id_columns', [])
        primary_keys = schema.get('primary_keys', [])

        # For inventory, use product_id and batch_number if available
        inventory_cols = schema.get('inventory_columns', {})
        if inventory_cols.get('batch_number'):
            batch_col = inventory_cols['batch_number']
            if batch_col in df.columns:
                product_col = None
                for col in df.columns:
                    if 'product' in col.lower() or 'item' in col.lower():
                        product_col = col
                        break
                if product_col:
                    key_columns = [product_col, batch_col]
                    df = df.drop_duplicates(subset=key_columns, keep='first')
                    self.last_report['duplicates_removed'] += int(before - len(df))
                    return df

        key_columns = id_columns or primary_keys or []

        if key_columns:
            # Keep first occurrence
            df = df.drop_duplicates(subset=key_columns, keep='first')
        else:
            # Remove complete duplicates
            df = df.drop_duplicates()

        self.last_report['duplicates_removed'] += int(before - len(df))
        return df

    def _add_metadata_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add metadata columns"""
        # Add row hash for deduplication
        # Add import date
        if '_import_date' not in df.columns:
            df['_import_date'] = datetime.now()
            self.last_report['metadata_columns_added'].append('_import_date')
        if '_data_version' not in df.columns:
            df['_data_version'] = '1.0'
            self.last_report['metadata_columns_added'].append('_data_version')

        return df

    def create_cleaning_pipeline(self, dataset_type: str) -> Dict[str, Any]:
        """Create a cleaning pipeline for specific dataset types"""
        pipelines = {
            'inventory_management': {
                'remove_duplicates': ['product_id', 'batch_number', 'warehouse'],
                'date_columns': ['stock_date', 'expiry_date', 'manufacture_date', 'restock_date'],
                'numeric_columns': ['stock_level', 'reorder_point', 'reorder_quantity', 'lead_time', 'turnover'],
                'categorical_columns': ['warehouse', 'batch_number', 'condition', 'supplier'],
                'inventory_specific': True
            },
            'sales_orders': {
                'remove_duplicates': ['order_id', 'product_id', 'customer_id'],
                'date_columns': ['order_date', 'ship_date'],
                'numeric_columns': ['sales', 'profit', 'discount', 'quantity'],
                'categorical_columns': ['segment', 'ship_mode', 'category', 'sub_category'],
                'inventory_specific': False
            },
            'transactions': {
                'remove_duplicates': ['transaction_id', 'customer_id'],
                'date_columns': ['transaction_date'],
                'numeric_columns': ['amount', 'fee', 'tax'],
                'categorical_columns': ['type', 'status'],
                'inventory_specific': False
            },
            'customer_data': {
                'remove_duplicates': ['customer_id', 'email'],
                'date_columns': ['birth_date', 'join_date'],
                'numeric_columns': ['age', 'income', 'score'],
                'categorical_columns': ['gender', 'segment', 'tier'],
                'inventory_specific': False
            }
        }

        return pipelines.get(dataset_type, {
            'remove_duplicates': [],
            'date_columns': [],
            'numeric_columns': [],
            'categorical_columns': [],
            'inventory_specific': False
        })
