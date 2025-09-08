# tools/classifier.py
from .mcp_registry import BaseTool, mcp_registry

class IntentClassifierTool(BaseTool):
    name = "intent_classifier"
    description = "Classifies a user's prompt to determine the correct domain agent (e.g., sales, finance, analytics)."

    def run(self, user_prompt: str) -> str:
        user_prompt_lower = user_prompt.lower()
        if any(k in user_prompt_lower for k in ["customer", "lead", "order", "sale", "crm"]):
            return "sales_agent"
        elif any(k in user_prompt_lower for k in ["report", "analytics", "data", "insights"]):
            return "analytics_agent"
        elif any(k in user_prompt_lower for k in ["invoice", "payment", "finance", "ledger"]):
            return "finance_agent"
        elif any(k in user_prompt_lower for k in ["inventory", "stock", "product", "reorder"]):
            return "inventory_agent"
        else:
            return "general_purpose"

mcp_registry.register(IntentClassifierTool())
