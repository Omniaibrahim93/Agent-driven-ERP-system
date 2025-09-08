# Dockerfile
# Use a slim Python image for a smaller footprint
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code, including the erp.db file
COPY . .

# Expose the ports for FastAPI (backend) and Streamlit (frontend)
EXPOSE 8000
EXPOSE 8501

# The command to run the application
# We'll use a single entrypoint script to start both the backend and frontend
CMD ["/bin/bash", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false"]
