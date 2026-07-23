import os

import pytest
from fastapi import HTTPException

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-for-explorer-scope-only-min-32-chars")

from api import _assert_explorer_query_is_tenant_scoped, _extract_table_refs  # noqa: E402
from models import TokenData  # noqa: E402


class DummyAgent:
    def _get_available_tables(self):
        return ["u7_sales", "u7_inventory", "u5_sales"]


def test_extract_table_refs_handles_joins_commas_subqueries_and_ctes():
    query = """
        WITH recent_sales AS (
            SELECT * FROM u7_sales WHERE order_date IS NOT NULL
        )
        SELECT *
        FROM recent_sales rs
        JOIN u7_inventory i ON i.sku = rs.sku
        WHERE EXISTS (SELECT 1 FROM u5_sales other WHERE other.sku = rs.sku)
    """

    assert _extract_table_refs(query) == {"u7_sales", "u7_inventory", "u5_sales"}


def test_explorer_allows_only_current_users_tables():
    user = TokenData(user_id=7, username="alice")

    _assert_explorer_query_is_tenant_scoped(
        "SELECT s.*, i.stock FROM u7_sales s JOIN u7_inventory i ON i.sku = s.sku",
        user,
        DummyAgent(),
    )


def test_explorer_rejects_another_users_table():
    user = TokenData(user_id=7, username="alice")

    with pytest.raises(HTTPException) as exc:
        _assert_explorer_query_is_tenant_scoped(
            "SELECT * FROM u5_sales",
            user,
            DummyAgent(),
        )

    assert exc.value.status_code == 403


def test_explorer_rejects_metadata_tables():
    user = TokenData(user_id=7, username="alice")

    with pytest.raises(HTTPException) as exc:
        _assert_explorer_query_is_tenant_scoped(
            "SELECT * FROM sqlite_master",
            user,
            DummyAgent(),
        )

    assert exc.value.status_code == 403
