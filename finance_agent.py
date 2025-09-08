# agents/finance_agent.py
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.tools import Tool as LangChainTool
from tools.mcp_registry import mcp_registry

llm = Ollama(model="llama3")

finance_prompt_template = """
You are a specialized financial agent. Your task is to automate financial operations such as invoice processing, ledger updates, and payment tracking.
You have access to the following tools:
{tools}

If the user wants to create, update, or delete financial data (such as invoices, payments), you must use the finance_sql_write tool.
If the user wants to retrieve data, you must use the finance_sql_read tool.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Conversation timeline:
{history}
Question: {input}
Thought: I need to analyze this request and determine what action to take.
Action: {agent_scratchpad}
"""

class FinanceAgent:
    def __init__(self):
        self.read_tool = mcp_registry.get_tool("finance_sql_read")
        self.write_tool = mcp_registry.get_tool("finance_sql_write")
        
        self.langchain_tools = [
            LangChainTool(name=self.read_tool.name, func=self.read_tool.run, description=self.read_tool.description),
            LangChainTool(name=self.write_tool.name, func=self.write_tool.run, description=self.write_tool.description)
        ]
        
        self.prompt = PromptTemplate.from_template(finance_prompt_template)
        self.agent = create_react_agent(llm, self.langchain_tools, self.prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=self.langchain_tools, verbose=True, handle_parsing_errors=True, max_iterations=10, max_execution_time=60)

    def run(self, user_prompt: str, memory_context: dict) -> str:
        return self.executor.invoke({"input": user_prompt, "history": memory_context.get("history", "")})['output']
