# Base image with Conda
FROM continuumio/miniconda3:latest

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy the Conda environment file first (for better Docker layer caching)
COPY environment.yml .

# Create the Conda environment and clean up to reduce image size
RUN conda env create -f environment.yml && \
    conda clean -afy

# Copy the full project into the container
COPY . .

# Activate the environment for all following commands
SHELL ["conda", "run", "-n", "recommender_env", "/bin/bash", "-c"]

# Expose port for Streamlit
EXPOSE 8501

# Health check to ensure the app is running properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch your Streamlit app avec chemin absolu
CMD ["conda", "run", "--no-capture-output", "-n", "recommender_env", "streamlit", "run", "/app/app/streamlitapp.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]