# tools/sales_sql.py
import sqlite3
import pandas as pd
from tools.mcp_registry import BaseTool, mcp_registry

DB_PATH = 'erp.db'

class SalesSQLReadTool(BaseTool):
    name = "sales_sql_read"
    description = "Executes read-only SQL queries on sales tables (customers, leads, orders, order_items)."

    def run(self, query: str) -> str:
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df.to_markdown(index=False)
        except sqlite3.Error as e:
            return f"SQL Error: {e}"

class SalesSQLWriteTool(BaseTool):
    name = "sales_sql_write"
    description = "Executes write SQL operations (INSERT, UPDATE, DELETE) on sales tables. Requires approval for sensitive actions."

    def run(self, query: str) -> str:
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            conn.close()
            return f"SQL write operation successful. Rows affected: {cursor.rowcount}"
        except sqlite3.Error as e:
            return f"SQL Error: {e}"

mcp_registry.register(SalesSQLReadTool())
mcp_registry.register(SalesSQLWriteTool())
