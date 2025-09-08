# agents/router_agent.py
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.tools import Tool as LangChainTool
from tools.mcp_registry import mcp_registry
from tools.classifier import IntentClassifierTool  # This will register the tool

llm = Ollama(model="llama3")

prompt_template = """
You are a central routing agent for an ERP system. Your task is to analyze a user's request and determine which specialized agent can best handle it.

You have access to the following tools:
{tools}

Available Agents:
- sales_agent: Handles customers, leads, and orders.
- analytics_agent: Provides insights and reports from data.
- finance_agent: Manages invoices and financial transactions.
- inventory_agent: Controls stock levels and supplier orders.

Use the tool to classify the request and then state the name of the appropriate agent.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
Thought: I need to use the intent_classifier tool to determine which agent should handle this request.
Action: intent_classifier
Action Input: {input}
Observation: {agent_scratchpad}
"""

class RouterAgent:
    def __init__(self):
        self.classifier_tool = mcp_registry.get_tool("intent_classifier")
        self.langchain_tool = LangChainTool(
            name=self.classifier_tool.name,
            func=self.classifier_tool.run,
            description=self.classifier_tool.description
        )
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.agent = create_react_agent(llm, [self.langchain_tool], self.prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=[self.langchain_tool], verbose=True, handle_parsing_errors=True, max_iterations=10, max_execution_time=60)

    def route_request(self, user_prompt: str) -> str:
        response = self.executor.invoke({"input": user_prompt})
        return response['output'].strip()