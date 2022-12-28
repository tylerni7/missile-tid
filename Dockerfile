FROM python:3.9

# Java needed for Spark API
RUN apt-get update -y \
    && apt-get install -y libcurl4-openssl-dev libgeos-dev

WORKDIR /tmp

COPY requirements-dev.txt .
COPY tid tid/
COPY .codecov.yml .
COPY pyproject.toml .

# Install python dependencies
RUN pip install -r requirements-dev.txt

ENTRYPOINT coverage run -m pytest -vv
