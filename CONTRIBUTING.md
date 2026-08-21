# Contributing

Use Python 3.12 and create a virtual environment.

```bash
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
pytest --cov=app --cov-fail-under=80
bandit -q -r app
pip-audit -r requirements.txt
```

Provider adapters must normalize their output into the contracts in `app/providers/base.py`.
Do not expose provider SDK response objects outside the adapter layer.
Do not add a new model directly to endpoint code. Add it to an alias chain and pricing catalog instead.
Never commit provider keys.
