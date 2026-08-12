#!/usr/bin/env bash
# Apply every pending Alembic migration to whatever DATABASE_URL currently
# resolves to (backend/config.py -> settings.DATABASE_URL, from .env or the
# real process environment).
#
# Alembic must be invoked with backend/ as the cwd, since alembic/env.py
# imports `config`, `database`, and `models.db_models` the same way
# main.py does. This script cd's there first so it can be run from anywhere.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

alembic upgrade head

# ---------------------------------------------------------------------------
# Other useful commands (run from this same directory, i.e. backend/):
#
#   Autogenerate a new revision from model changes:
#     alembic revision --autogenerate -m "describe the change"
#
#   Roll back the most recent migration:
#     alembic downgrade -1
#
#   Roll back everything (drops all Eka tables):
#     alembic downgrade base
#
#   Check current DB revision vs. head:
#     alembic current
#     alembic heads
