Project Overview
This project delivers a modern, agent-driven Enterprise Resource Planning (ERP) system for Helios Dynamics, a rapidly expanding manufacturing and distribution firm. The system is designed to replace rigid, traditional ERP workflows with a flexible, intelligent, and narrative-based interface.

The core of the system is a multi-agent architecture where specialized agents for Sales, Analytics, Finance, and Inventory handle specific business tasks. A central Router Agent acts as the orchestrator, intelligently directing natural language requests from users to the appropriate domain-specific agent.

Key Features
Chat-Based Workflows: Interact with the ERP system using a natural language chat interface.

Modular Architecture: The system is built on a "Modular Composable Protocols (MCP)" framework, allowing for easy addition of new agents and tools.

Local-First Stack: Utilizes local components like ollama and SQLite to ensure data privacy, reduce costs, and enable rapid prototyping.

Explainable Insights: The Analytics Agent provides not just data, but also narrative explanations for results using a Text-to-SQL approach.

Unified Data Core: All agents operate on a single erp.db (SQLite) database, ensuring data consistency across departments.

Containerized Deployment: The entire application is packaged in a Docker container for consistent, cross-platform execution.

System Architecture
The project is structured into three main layers:

Frontend (Streamlit): A user-friendly chat interface that sends user prompts to the backend.

Backend (FastAPI): An API gateway that receives user requests, routes them to the correct agent, and returns a response.

Agents & Tools: The core logic of the system, including the Router Agent and specialized domain agents, each with their own set of tools to interact with the database.

Agent Responsibilities
Router Agent: The central brain of the system. It uses an intent classifier to analyze user prompts and route the request to the correct domain agent (Sales, Analytics, Finance, or Inventory).

Sales Agent: Manages customers, leads, orders, and sales inquiries by performing read and write operations on the sales database tables.

Analytics Agent: Answers quantitative questions and provides data-driven insights by converting natural language into SQL queries.

Finance Agent: Automates financial workflows, including invoice processing and ledger updates, by interacting with financial tables.

Inventory Agent: Maintains and monitors inventory levels, tracks products, and manages purchase orders.

Tool Integration
All agents are built on a "Modular Composable Protocols (MCP)" system using a centralized ToolRegistry. This registry allows for a plug-and-play architecture where tools can be easily registered and accessed by different agents, ensuring high modularity and testability.

Getting Started
Follow these steps to set up and run the application locally.

Prerequisites
Python 3.9+: Ensure Python is installed on your system.

Docker Desktop: Required for containerized deployment.

Ollama: A local LLM server. Download and install it, then pull the required model:

ollama pull llama3

Local Execution
Clone the Repository:

git clone <your_repository_url>
cd agent-driven-erp

Run with Docker Compose:
With Docker Desktop running, execute a single command from the project's root directory:

docker-compose up --build

This command will build the Docker image, start the backend (FastAPI) and frontend (Streamlit) services, and get the entire system up and running.

Access the Application:
Open your web browser and navigate to http://localhost:8501 to access the chat interface.

Sample Queries
Here are a few examples to test the system and see each agent in action:

Sales Agent:

show all customers

create a new lead with a name 'Ahmed' and email 'ahmed@example.com'

Analytics Agent:

what is the total revenue from all orders?

explain the term 'sales pipeline'

Finance Agent:

list all invoices

create a new invoice for customer with ID 1

Inventory Agent:

show me the current stock levels for all products

add 10 units of product 'P-101' to stock
