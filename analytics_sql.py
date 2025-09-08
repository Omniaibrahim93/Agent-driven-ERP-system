# tools/analytics_sql.py
import sqlite3
import pandas as pd
from tools.mcp_registry import BaseTool, mcp_registry
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

DB_PATH = 'erp.db'
llm = Ollama(model="llama3")

class TextToSQLTool(BaseTool):
    name = "text_to_sql_tool"
    description = "Converts a natural language question into a SQLite SQL query and executes it."
    
    sql_prompt = PromptTemplate.from_template("""
    Given the database schema below, write a concise, valid SQLite SQL query that answers the user's question.
    Tables: customers, orders, order_items, invoices, payments, glossary, stock.
    Question: {question}
    SQL Query: 
    """)

    def run(self, user_question: str) -> str:
        try:
            chain = self.sql_prompt | llm
            sql_query = chain.invoke({"question": user_question}).strip().replace("```sql", "").replace("```", "")
            
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            return df.to_markdown(index=False)
        except Exception as e:
            return f"Error translating or executing SQL: {e}"

class GlossaryReadTool(BaseTool):
    name = "glossary_read"
    description = "Retrieves the definition of a specific business term from the glossary table."

    def run(self, term: str) -> str:
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT definition FROM glossary WHERE term = ?", (term,))
            result = cursor.fetchone()
            conn.close()
            return result[0] if result else f"Term '{term}' not found."
        except sqlite3.Error as e:
            return f"SQL Error: {e}"

mcp_registry.register(TextToSQLTool())
mcp_registry.register(GlossaryReadTool())
