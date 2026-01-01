python -m src.training.train_all

python -m http.server 5500

python -m uvicorn api.main:app --reload --port 8000