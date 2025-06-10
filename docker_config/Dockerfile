# ─── Base & System Dependencies ──────────────────────────────────────────────
FROM python:3.10-slim-bookworm AS base

# Install OS packages for .NET, DWSIM runtime libs, and common Python data tooling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl gnupg apt-transport-https \
      libgl1 libsm6 libxrender1 libfontconfig1 \
      build-essential \
      mono-complete \
    && rm -rf /var/lib/apt/lists/*

# ─── .NET SDK for pythonnet / DWSIM ───────────────────────────────────────────
RUN curl -sSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin --channel 8.0
ENV DOTNET_ROOT=/root/.dotnet
ENV PATH=$PATH:/root/.dotnet:/root/.dotnet/tools

# ─── Set up DWSIM DLLs ────────────────────────────────────────────────────────
# Create DWSIM libs directory
RUN mkdir -p /opt/dwsim_libs
ENV DWSIM_DLL_PATH=/opt/dwsim_libs

# Copy DWSIM DLLs
COPY dwsim_libs/*.dll /opt/dwsim_libs/

# ─── Python Dependencies ─────────────────────────────────────────────────────
WORKDIR /app
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install pythonnet for CLR support
RUN pip install --no-cache-dir pythonnet

# ─── Copy Your Code ──────────────────────────────────────────────────────────
# Copy all source, scripts, and config files
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Create config directory if it doesn't exist
RUN mkdir -p ./config

# ─── Entrypoint ─────────────────────────────────────────────────────────────
# Replace `run_intel_system.py` with your actual main script
ENTRYPOINT ["python", "scripts/run_intel_system.py"] 