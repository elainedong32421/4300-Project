# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-build

WORKDIR /app/frontend

COPY frontend/package*.json ./

RUN npm install

COPY frontend/ ./

RUN npm run build

# Stage 2: Setup Python backend
FROM python:3.10-slim

ENV CONTAINER_HOME=/var/www

WORKDIR $CONTAINER_HOME

COPY requirements.txt $CONTAINER_HOME/requirements.txt
RUN pip install --no-cache-dir -r $CONTAINER_HOME/requirements.txt

COPY src/ $CONTAINER_HOME/src/

COPY --from=frontend-build /app/frontend/dist $CONTAINER_HOME/frontend/dist

CMD ["gunicorn", "--chdir", "src", "app:app", "--bind", "0.0.0.0:5000"]
