"""
Step 2 — ETL: load structured Excel sheets into Postgres.

Sheets:
  - Investments_data_50k  -> investments
  - performance_data      -> performance

Usage:
  python -m etl.load_structured
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql

# Allow `python -m etl.load_structured` from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import XLSX_PATH, SCHEMA_STRUCTURED_SQL, database_url  # noqa: E402


INVESTMENT_COLUMNS = {
    "Client_Id__c": "client_id",
    "Client Name": "client_name",
    "Client_Group_Id_c": "client_group_id",
    "deal_id": "deal_id",
    "id_CapitalCall": "capital_call_id",
    "Investment_Amount_USD_for_agg": "investment_amount_usd",
    "Investment_Amount_Natural_Currency": "investment_amount_natural",
    "Natural_Currency_Code": "natural_currency_code",
    "Investment_Exchange_Rate": "investment_exchange_rate",
    "dat_MinInvested": "invested_date",
    "deal_name": "deal_name",
    "cod_lob": "lob_code",
    "nam_lob": "lob_name",
    # "client_id" in Excel is a duplicate of Client_Id__c — skip
    "flg_Realised": "realised",
    "ClientStatus": "client_status",
    "AccountName": "account_name",
    "AccountName_org": "account_name_org",
    "AccountRM": "account_rm",
    "AccountRM_org": "account_rm_org",
    "AccountOwnerId": "account_owner_id",
    "AccountRMEmail": "account_rm_email",
    "AccountRMEmail_org": "account_rm_email_org",
    "RM_Alias": "rm_alias",
}

PERFORMANCE_COLUMNS = {
    "Client_Group_Id": "client_group_id",
    "Client_Id": "client_id",
    "Client_Name": "client_name",
    "IsGroup_Flag": "is_group",
    "CI_CY_FR_Amount": "ci_cy_fr_amount",
    "CI_Current_IRR": "ci_current_irr",
    "CI_FR_SI_Amount": "ci_fr_si_amount",
    "CI_Current_MOIC": "ci_current_moic",
    "CI_L3Y_DIS_Amount": "ci_l3y_dis_amount",
    "CI_Total_IRR": "ci_total_irr",
    "CI_L3Y_FR_Amount": "ci_l3y_fr_amount",
    "CI_Total_MOIC": "ci_total_moic",
    "HF_AUM_Amount": "hf_aum_amount",
    "CI_Realised_IRR": "ci_realised_irr",
    "Mena_AUM_Amount": "mena_aum_amount",
    "CI_Realised_MOIC": "ci_realised_moic",
    "Pref_Shares_AUM_Amount": "pref_shares_aum_amount",
    "CI_Since_2001_IRR": "ci_since_2001_irr",
    "RE_CY_FR_Amount": "re_cy_fr_amount",
    "HF_Total_MOIC": "hf_total_moic",
    "RE_FR_SI_Amount": "re_fr_si_amount",
    "HF_Total_IRR": "hf_total_irr",
    "RE_L3Y_DIS_Amount": "re_l3y_dis_amount",
    "RE_Core_IRR": "re_core_irr",
    "RE_L3Y_FR_Amount": "re_l3y_fr_amount",
    "RE_Core_MOIC": "re_core_moic",
    "RE_AUM_Amount": "re_aum_amount",
    "RE_Current_MOIC": "re_current_moic",
    "Receivables_Amount": "receivables_amount",
    "RE_Current_IRR": "re_current_irr",
    "Tech_AUM_Amount": "tech_aum_amount",
    "RE_Total_IRR": "re_total_irr",
    "As_Of_Date": "as_of_date",
    "RE_Total_MOIC": "re_total_moic",
    "Client_Last_Met_Date": "client_last_met_date",
    "RE_Realised_IRR": "re_realised_irr",
    "Last_HF_Investment_Date": "last_hf_investment_date",
    "RE_Realised_MOIC": "re_realised_moic",
    "Last_CI_Investment_Date": "last_ci_investment_date",
    "Last_CI_Investment_Name": "last_ci_investment_name",
    "Last_CI_Investment_Amount": "last_ci_investment_amount",
    "Last_RE_Investment_Name": "last_re_investment_name",
    "Last_RE_Investment_Date": "last_re_investment_date",
    "First_CI_Investment_Name": "first_ci_investment_name",
    "Last_RE_Investment_Amount": "last_re_investment_amount",
    "First_RE_Investment_Name": "first_re_investment_name",
    "First_HF_Investment_Date": "first_hf_investment_date",
    "Investment_Status_Name": "investment_status_name",
    "First_CI_Investment_Date": "first_ci_investment_date",
    "CI_Status_Name": "ci_status_name",
    "First_RE_Investment_Date": "first_re_investment_date",
    "RE_Status_Name": "re_status_name",
    "Call_Account_Balance_Amount": "call_account_balance_amount",
    "INF_Status_Name": "inf_status_name",
    "Product_Count_Number": "product_count_number",
    "ICM_Status_Name": "icm_status_name",
    "Total_AUM_Amount": "total_aum_amount",
    "Future_Distribution_Amount": "future_distribution_amount",
    "CI_AUM_Amount": "ci_aum_amount",
    "COP_Total_IRR": "cop_total_irr",
    "Last_COP_Investment_Name": "last_cop_investment_name",
    "COP_Realised_IRR": "cop_realised_irr",
    "COP_Current_IRR": "cop_current_irr",
    "COP_FR_SI_Amount": "cop_fr_si_amount",
    "Last_COP_Investment_Amount": "last_cop_investment_amount",
}


def apply_schema(conn: psycopg.Connection) -> None:
    ddl = SCHEMA_STRUCTURED_SQL.read_text()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    print(f"Applied schema from {SCHEMA_STRUCTURED_SQL.name}")


def _to_python(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    return value


def copy_dataframe(conn: psycopg.Connection, table: str, df: pd.DataFrame) -> int:
    """Bulk-load via COPY FROM STDIN (CSV)."""
    cols = list(df.columns)
    # Normalize NaN -> None for CSV
    export = df.copy()
    for c in export.columns:
        if pd.api.types.is_datetime64_any_dtype(export[c]):
            export[c] = export[c].dt.strftime("%Y-%m-%d")
    export = export.where(pd.notnull(export), None)

    col_list = sql.SQL(", ").join(sql.Identifier(c) for c in cols)
    copy_sql = sql.SQL(
        "COPY {} ({}) FROM STDIN WITH (FORMAT CSV, NULL '')"
    ).format(sql.Identifier(table), col_list)

    buf_rows = []
    for row in export.itertuples(index=False, name=None):
        cells = []
        for v in row:
            if v is None:
                cells.append("")
            else:
                s = str(v)
                # Escape CSV special chars
                if any(ch in s for ch in [",", '"', "\n", "\r"]):
                    s = '"' + s.replace('"', '""') + '"'
                cells.append(s)
        buf_rows.append(",".join(cells))
    payload = "\n".join(buf_rows) + ("\n" if buf_rows else "")

    with conn.cursor() as cur:
        with cur.copy(copy_sql) as copy:
            copy.write(payload.encode("utf-8"))
    conn.commit()
    return len(df)


def load_investments(conn: psycopg.Connection) -> int:
    print("Reading Investments_data_50k …")
    df = pd.read_excel(XLSX_PATH, sheet_name="Investments_data_50k")
    # Select source cols first so Excel's duplicate `client_id` is dropped
    # (we keep Client_Id__c → client_id).
    df = df[list(INVESTMENT_COLUMNS.keys())].rename(columns=INVESTMENT_COLUMNS)

    df["invested_date"] = pd.to_datetime(df["invested_date"], errors="coerce")
    df["client_group_id"] = pd.to_numeric(df["client_group_id"], errors="coerce").astype("Int64")
    df["capital_call_id"] = pd.to_numeric(df["capital_call_id"], errors="coerce").astype("Int64")
    # Int64 -> object with None for COPY
    for col in ("client_group_id", "capital_call_id"):
        df[col] = df[col].astype(object).where(df[col].notna(), None)

    print(f"Loading {len(df):,} investment rows …")
    n = copy_dataframe(conn, "investments", df)
    print(f"  → investments: {n:,} rows")
    return n


def load_performance(conn: psycopg.Connection) -> int:
    print("Reading performance_data …")
    df = pd.read_excel(XLSX_PATH, sheet_name="performance_data")
    df = df[list(PERFORMANCE_COLUMNS.keys())].rename(columns=PERFORMANCE_COLUMNS)

    # Y/N → boolean
    df["is_group"] = df["is_group"].map(
        lambda x: True if str(x).strip().upper() == "Y" else False if str(x).strip().upper() == "N" else None
    )

    date_cols = [
        "as_of_date",
        "client_last_met_date",
        "last_hf_investment_date",
        "last_ci_investment_date",
        "last_re_investment_date",
        "first_hf_investment_date",
        "first_ci_investment_date",
        "first_re_investment_date",
    ]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    df["client_group_id"] = pd.to_numeric(df["client_group_id"], errors="coerce").astype("Int64")
    df["product_count_number"] = pd.to_numeric(df["product_count_number"], errors="coerce").astype("Int64")
    for col in ("client_group_id", "product_count_number"):
        df[col] = df[col].astype(object).where(df[col].notna(), None)

    print(f"Loading {len(df):,} performance rows …")
    n = copy_dataframe(conn, "performance", df)
    print(f"  → performance: {n:,} rows")
    return n


def verify(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM investments")
        inv = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM performance")
        perf = cur.fetchone()[0]
        cur.execute(
            "SELECT client_id, deal_name, investment_amount_usd "
            "FROM investments ORDER BY investment_amount_usd DESC NULLS LAST LIMIT 3"
        )
        top = cur.fetchall()
        cur.execute(
            "SELECT client_id, is_group, total_aum_amount "
            "FROM performance WHERE is_group = false "
            "ORDER BY total_aum_amount DESC NULLS LAST LIMIT 3"
        )
        top_aum = cur.fetchall()
    print("\nVerification")
    print(f"  investments count : {inv:,}")
    print(f"  performance count : {perf:,}")
    print("  top investments   :", top)
    print("  top client AUM    :", top_aum)


def main() -> None:
    if not XLSX_PATH.exists():
        raise FileNotFoundError(f"Missing Excel file: {XLSX_PATH}")

    url = database_url()
    print(f"Connecting to {url.split('@')[-1].split('?')[0]} …")
    with psycopg.connect(url, connect_timeout=30) as conn:
        apply_schema(conn)
        load_investments(conn)
        load_performance(conn)
        verify(conn)
    print("\nDone — structured data loaded.")


if __name__ == "__main__":
    main()
