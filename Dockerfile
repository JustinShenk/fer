FROM python:3.7-slim

LABEL description="Test"

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ARG ENVIRONMENT
ENV ENVIRONMENT=${ENVIRONMENT}

RUN mkdir -p /usr/share/man/man1 \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgtk2.0-dev\
    libglib2.0-0\
    ffmpeg\
    libsm6\
    libxext6\
    git\
    wget\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache/*

COPY . /srv/testing
WORKDIR /srv/testing

# set environment variable
ENV PYTHONDONTWRITEBYTECODE 1

EXPOSE 8000
CMD ["bash", "run.sh"]
