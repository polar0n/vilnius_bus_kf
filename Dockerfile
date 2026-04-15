# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install necessary system dependencies
# Removed software-properties-common as it is not needed for this environment
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Configure port and execution for Cloud Run
EXPOSE 8080
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]