"""
Alembic environment for the General WAR project.

The database URL is not stored in alembic.ini. It is resolved through
``pipeline.db.database_url``, so migrations always target the same database
the pipeline uses, and no credential is written to a tracked file.

There is no SQLAlchemy metadata to autogenerate against: the schema is
authored as SQL in ``config/schema.sql`` and the project uses SQLAlchemy Core
rather than the ORM. ``target_metadata`` is therefore None, and migrations
are written by hand.
"""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from pipeline.db import database_url

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Hand-written migrations only; see the module docstring.
target_metadata = None


def _url() -> str:
    """Resolve the database URL for this migration run.

    Returns:
        The URL from DATABASE_URL, or from an explicit ``-x url=...`` override.

    Raises:
        DatabaseConfigError: If neither is set.
    """
    # `alembic -x url=...` lets a one-off migration target another database
    # without exporting DATABASE_URL for the whole shell.
    override = context.get_x_argument(as_dictionary=True).get("url")
    return database_url(override)


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a database."""
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    section = config.get_section(config.config_ini_section, {}) or {}
    section["sqlalchemy.url"] = _url()

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
