FROM python:3.12-slim

# Install git for cloning repositories
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Build arguments provided by the Git Escrows system
ARG SOURCE_REPO
ARG SOURCE_COMMIT
ARG TEST_REPO
ARG TEST_COMMIT

# Create directory structure
WORKDIR /app

# Clone the solution repository (the candidate's submission)
RUN git clone ${SOURCE_REPO} solution && \
    cd solution && \
    git checkout ${SOURCE_COMMIT}

# Clone the test repository (this repository)
RUN git clone ${TEST_REPO} tests && \
    cd tests && \
    git checkout ${TEST_COMMIT}

# Set up the test environment
WORKDIR /app/tests

# Install dependencies using Poetry
# We disable virtualenvs to install directly into the container environment
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Set PYTHONPATH to include the solution directory so tests can import it
# Adjust this if the solution code is in a subdirectory (e.g., /app/solution/src)
ENV PYTHONPATH="/app/solution:$PYTHONPATH"

# Default command to run tests
CMD ["poetry", "run", "pytest"]
