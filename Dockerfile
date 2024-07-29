FROM node:16.14.2
WORKDIR /app

COPY /flask_dropzone /app/flask_dropzone
COPY /guide /app/guide
COPY /model /app/model
COPY /static /app/static
COPY /templates /app/templates
COPY /upload /app/upload

COPY .git /app/.git
COPY .dockerignore /app/.dockerignore
COPY .gitignore /app/.gitignore
COPY app.py /app/app.py
COPY docker-compose.yaml /app/docker-compose.yaml
COPY notes.txt /app/notes.txt
COPY tailwind.config.js /app/tailwind.config.js
COPY utils.py /app/utils.py

COPY Dockerfile ./
COPY package.json package.json
COPY package-lock.json package-lock.json


# Install Node.js
RUN apt-get update && apt-get install -y curl
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -
RUN apt-get install -y nodejs
RUN npm ci
RUN npm run build

# Install Python
FROM python:3.9.7

WORKDIR /app

COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "/app/app.py"]

