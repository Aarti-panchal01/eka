"""Alembic environment for Eka.

Must be run with cwd = backend/ (or invoked via run_migrations.sh, which cds
there for you) since `config` and `database` are imported as top-level
modules the same way main.py imports them.
"""

import asyncio
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# --------------------------------------------------------------------------
# Make `backend/` importable regardless of the caller's cwd, so
# `from config import settings` etc. below resolve the same way they do for
# main.py. alembic/env.py -> alembic/ -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import settings  # noqa: E402
from database import Base  # noqa: E402
import models.db_models  # noqa: E402,F401  (registers all tables on Base.metadata)

# this is the Alembic Config object, which provides access to the values
# within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here for 'autogenerate' support
target_metadata = Base.metadata

# --------------------------------------------------------------------------
# The real DB URL lives in settings.DATABASE_URL (already normalised to
# postgresql+asyncpg://... by config.py), not in alembic.ini.
#
# configparser (which alembic.ini is parsed with) treats `%` as the start of
# an interpolation token, e.g. `%(foo)s`. A password containing a literal `%`
# would otherwise raise `InterpolationSyntaxError` or get silently mangled.
# Escape every `%` as `%%` before handing the URL to set_main_option, which
# runs it back through the same interpolation-aware config parser.
_db_url = settings.DATABASE_URL.replace("%", "%%")
config.set_main_option("sqlalchemy.url", _db_url)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This generates SQL scripts without a live DB connection. Uses the
    unescaped URL directly (no configparser round-trip involved here).
    """
    url = settings.DATABASE_URL
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an Engine and associate a connection with the context."""
    configuration = config.get_section(config.config_ini_section, {})

    connect_args: dict = {}
    # Supabase's pgbouncer pooler (port 6543, or a URL containing "pooler")
    # runs in transaction-pooling mode, which is incompatible with asyncpg's
    # default use of server-side prepared statements: a statement prepared on
    # one physical connection can be replayed against a different one by the
    # pooler and blow up with "prepared statement does not exist". Disabling
    # the statement cache makes asyncpg fall back to unnamed/unprepared
    # statements, which is safe under transaction pooling.
    if "pooler" in settings.DATABASE_URL or ":6543" in settings.DATABASE_URL:
        connect_args["statement_cache_size"] = 0

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode using an async engine."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
