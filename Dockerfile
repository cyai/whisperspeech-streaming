# Use an official Python runtime as a parent image
FROM python:3.10.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1

# Install necessary packages
RUN apt-get update && \
    apt-get install -y lsof unzip llvm build-essential python3-dev git wget curl && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    apt-add-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main" && \
    apt-get update && \
    wget https://mirrors.kernel.org/ubuntu/pool/main/libf/libffi/libffi6_3.2.1-8_amd64.deb && \
    apt install ./libffi6_3.2.1-8_amd64.deb && \
    apt-get install -y llvm-9 && \
    rm -rf /var/lib/apt/lists/*

# Clone the repository
RUN git clone https://github.com/cyai/whisperspeech-streaming /app

# Set the working directory
WORKDIR /app

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    source $HOME/.cargo/env

# Install Poetry
RUN pip install poetry

# Install project dependencies
RUN poetry install --no-root --no-interaction --no-ansi

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["poetry", "run", "uvicorn", "whisperspeech_streaming_server.main:app", "--host", "0.0.0.0", "--port", "8000"]