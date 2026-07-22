import os
import time
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-for-smoke-tests-only-min-32-chars")

from api import app  # noqa: E402


ROOT = Path(__file__).resolve().parent


def _signup_and_token(client: TestClient, username: str) -> str:
    response = client.post(
        "/api/auth/signup",
        json={
            "username": username,
            "email": f"{username}@example.com",
            "password": "StrongPass123!",
            "confirm_password": "StrongPass123!",
        },
    )
    if response.status_code == 400 and "already" in response.text.lower():
        response = client.post(
            "/api/auth/login",
            json={"username": username, "password": "StrongPass123!"},
        )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _upload_dataset(client: TestClient, token: str, csv_name: str, table_name: str):
    with (ROOT / csv_name).open("rb") as file:
        response = client.post(
            "/api/upload",
            headers={"Authorization": f"Bearer {token}"},
            data={"table_name": table_name},
            files={"file": (csv_name, file, "text/csv")},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["rows"] > 0
    assert payload["columns"]
    assert "cleaning_report" in payload
    return payload


def _assert_basic_dataset_flow(client: TestClient, token: str, expected_table: str):
    headers = {"Authorization": f"Bearer {token}"}

    tables = client.get("/api/tables", headers=headers)
    assert tables.status_code == 200, tables.text
    assert expected_table in tables.json()["tables"]

    schema = client.get("/api/schema", headers=headers)
    assert schema.status_code == 200, schema.text
    schema_payload = schema.json()
    assert schema_payload["row_count"] > 0
    assert schema_payload["numeric_columns"] or schema_payload["string_columns"]

    kpi = client.get("/api/kpi/all", headers=headers)
    assert kpi.status_code == 200, kpi.text
    assert {"revenue", "profit", "orders", "customers"}.issubset(kpi.json())

    me = client.get("/api/auth/me", headers=headers)
    assert me.status_code == 200, me.text
    actual_table = f"u{me.json()['user_id']}_{expected_table}"
    explorer = client.post(
        "/api/explorer",
        headers=headers,
        json={"query": f"SELECT COUNT(*) AS row_count FROM {actual_table}"},
    )
    assert explorer.status_code == 200, explorer.text
    assert explorer.json()["count"] == 1

    return schema_payload, kpi.json()


def test_sample_superstore_project_flow():
    client = TestClient(app)
    username = f"smoke_superstore_{int(time.time())}"
    table_name = f"superstore_smoke_{int(time.time())}"
    token = _signup_and_token(client, username)

    upload = _upload_dataset(client, token, "Sample_Superstore.csv", table_name)
    assert upload["cleaning_report"]["final_rows"] == upload["rows"]

    schema, kpi = _assert_basic_dataset_flow(client, token, table_name)
    assert schema["dataset_type"] in {"sales_orders", "general", "transactions"}
    assert kpi["orders"] >= 0


def test_dummy_inventory_project_flow():
    client = TestClient(app)
    username = f"smoke_inventory_{int(time.time())}"
    table_name = f"inventory_smoke_{int(time.time())}"
    token = _signup_and_token(client, username)
    headers = {"Authorization": f"Bearer {token}"}

    upload = _upload_dataset(client, token, "dummy_inventory_8000.csv", table_name)
    assert upload["cleaning_report"]["final_rows"] == upload["rows"]

    schema, _ = _assert_basic_dataset_flow(client, token, table_name)
    assert schema["dataset_type"] == "inventory_management"
    assert schema["inventory_columns"]

    metrics = client.get("/api/inventory/metrics", headers=headers)
    assert metrics.status_code == 200, metrics.text
    metrics_payload = metrics.json()
    assert {"total_stock", "inventory_value", "low_stock", "out_of_stock"}.issubset(metrics_payload)

    analysis_types = [
        "stock_status",
        "inventory_value",
        "turnover_analysis",
        "reorder_analysis",
        "supplier_analysis",
        "warehouse_analysis",
        "expiry_analysis",
        "abc_classification",
    ]
    for analysis_type in analysis_types:
        response = client.post(
            "/api/inventory/analysis",
            headers=headers,
            json={"analysis_type": analysis_type, "criteria": "value"},
        )
        assert response.status_code == 200, f"{analysis_type}: {response.text}"
        payload = response.json()
        assert "columns" in payload and "rows" in payload
        assert payload["rows"], f"{analysis_type}: {payload.get('message')}"


def test_dataset_visibility_and_duplicate_filename_are_per_user():
    client = TestClient(app)
    stamp = int(time.time())
    user_a = f"visibility_a_{stamp}"
    user_b = f"visibility_b_{stamp}"
    token_a = _signup_and_token(client, user_a)
    token_b = _signup_and_token(client, user_b)
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    _upload_dataset(client, token_a, "Sample_Superstore.csv", f"a_superstore_{stamp}")
    user_a_id = client.get("/api/auth/me", headers=headers_a).json()["user_id"]
    actual_a_table = f"u{user_a_id}_a_superstore_{stamp}"

    tables_a = client.get("/api/tables", headers=headers_a)
    tables_b = client.get("/api/tables", headers=headers_b)
    assert tables_a.status_code == 200, tables_a.text
    assert tables_b.status_code == 200, tables_b.text
    assert f"a_superstore_{stamp}" in tables_a.json()["tables"]
    assert f"a_superstore_{stamp}" not in tables_b.json()["tables"]

    with (ROOT / "Sample_Superstore.csv").open("rb") as file:
        duplicate_for_same_user = client.post(
            "/api/upload",
            headers=headers_a,
            data={"table_name": f"a_superstore_duplicate_{stamp}"},
            files={"file": ("Sample_Superstore.csv", file, "text/csv")},
        )
    assert duplicate_for_same_user.status_code == 409

    _upload_dataset(client, token_b, "Sample_Superstore.csv", f"b_superstore_{stamp}")
    user_b_id = client.get("/api/auth/me", headers=headers_b).json()["user_id"]
    actual_b_table = f"u{user_b_id}_b_superstore_{stamp}"
    tables_b_after = client.get("/api/tables", headers=headers_b)
    assert tables_b_after.status_code == 200, tables_b_after.text
    assert f"b_superstore_{stamp}" in tables_b_after.json()["tables"]
    assert f"a_superstore_{stamp}" not in tables_b_after.json()["tables"]

    own_explorer = client.post(
        "/api/explorer",
        headers=headers_b,
        json={"query": f"SELECT COUNT(*) AS row_count FROM {actual_b_table}"},
    )
    assert own_explorer.status_code == 200, own_explorer.text

    cross_tenant_explorer = client.post(
        "/api/explorer",
        headers=headers_b,
        json={"query": f"SELECT COUNT(*) AS row_count FROM {actual_a_table}"},
    )
    assert cross_tenant_explorer.status_code == 403
