# Big Sur Marathon Simulator -- Streamlit container
# Build:  docker build -t bigsur-sim .
# Run:    docker run -p 8501:8501 bigsur-sim
# Open:   http://localhost:8501

FROM python:3.11-slim

WORKDIR /app

# System deps for numpy/scipy/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy simulation code + Streamlit UI
COPY big_sur_simpy.py big_sur_visuals.py app.py ./

EXPOSE 8501

# Streamlit runs on 0.0.0.0:8501 inside container; host port mapped via -p
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
