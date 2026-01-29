# Install

## Requirements
- Python 3.11
- Pinned versions are listed in `requirements-lock.txt`.

## Editable install
```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

## Verify
```bash
pytest
```

## Scripts
Use the `scripts/` modules for training, evaluation, and toy environment runs.
