# Base Image: Python 3.10+ (Slim for smaller size)
FROM python:3.10-slim

# Working Directory
WORKDIR /app

# System Dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Project Files
COPY pyproject.toml .

# 1. Copy the Vendor Directory (The Quantum Engine)
# This assumes you have copied the "QuantumSimulation" folder into "Project AEON/vendor"
COPY vendor/ ./vendor/

# 2. Copy the AEON Source Code
COPY src/ ./src/
COPY README.md .

# 3. Install Dependencies
# First, install the bundled Quantum Engine (Vendor Strategy)
# We use --no-deps to avoid re-installing numpy if we want to control it, 
# but allowing deps is usually safer for MVP.
RUN pip install ./vendor

# Then install AEON and its standard dependencies
RUN pip install .
RUN pip install fastapi uvicorn python-multipart python-dotenv

# Environment Variables
ENV ANTHROPIC_API_KEY=""
ENV AEON_CORS_ORIGINS="https://bluerose.systems,http://localhost:3000"

# Port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
