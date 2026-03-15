# PhishGuard AI - Engineering Automation

.PHONY: install train run docker-build docker-run clean help

help:
	@echo "PhishGuard AI - Available Commands:"
	@echo "  make install      Install all dependencies"
	@echo "  make train        Fine-tune DistilBERT model (requires GPU/MPS)"
	@echo "  make run          Start the Flask API and React Dashboard"
	@echo "  make docker-build Build Docker containers for production"
	@echo "  make docker-run   Run the full stack via Docker Compose"
	@echo "  make clean        Remove temporary files and caches"

install:
	pip install -r requirements.txt
	cd frontend && npm install

train:
	python src/train.py

run:
	@echo "Starting backend..."
	python app.py &
	@echo "Starting frontend..."
	cd frontend && npm start

docker-build:
	docker-compose build

docker-run:
	docker-compose up

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleaned up temporary files."