.PHONY: setup train serve test lint promote clean

PYTHON := python
PYTEST := pytest
COVERAGE_MIN := 80

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Downloading dataset..."
	$(PYTHON) -m data.ingest
	@echo "Setup complete."

# ── Training ──────────────────────────────────────────────────────────────────
train:
	@echo "Starting training pipeline..."
	PYTHONPATH=. $(PYTHON) -m models.train
	@echo "Training complete. Open http://localhost:5000 to view MLflow runs."

# ── Serving ───────────────────────────────────────────────────────────────────
serve:
	@echo "Starting all services (API + Dashboard + MLflow + Airflow)..."
	docker compose up --build -d
	@echo ""
	@echo "Services running:"
	@echo "  MLflow UI   → http://localhost:5000"
	@echo "  API         → http://localhost:8000/docs"
	@echo "  Dashboard   → http://localhost:8501"
	@echo "  Airflow     → http://localhost:8080"

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	PYTHONPATH=. $(PYTEST) tests/ \
		--cov=api --cov=models --cov=data \
		--cov-report=term-missing \
		--cov-fail-under=$(COVERAGE_MIN) \
		-v

# ── Linting ───────────────────────────────────────────────────────────────────
lint:
	ruff check .
	ruff format --check .
	mypy api/ models/ data/ --ignore-missing-imports --no-error-summary

lint-fix:
	ruff check . --fix
	ruff format .

# ── Model promotion ───────────────────────────────────────────────────────────
promote:
	@echo "Running promotion check (Staging → Production)..."
	PYTHONPATH=. $(PYTHON) -m models.registry

# ── Drift simulation ──────────────────────────────────────────────────────────
simulate-drift:
	@echo "Generating 7-day drift history snapshots..."
	PYTHONPATH=. $(PYTHON) -c "
from data.ingest import load_raw_dataset
from data.simulate_drift import generate_drift_history
generate_drift_history(load_raw_dataset())
"
	@echo "Drift snapshots saved to data/processed/drift_history/"

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	@echo "Removing temporary artefacts..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache  -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ coverage.xml
	@echo "Clean complete."

down:
	docker compose down -v
