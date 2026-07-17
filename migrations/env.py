"""
Alembic Migration Environment Configuration
=============================================
This file is executed by Alembic when running migration commands.
It configures SQLAlchemy, loads our app's Settings (to get the real
database URL from .env), and imports all models so Alembic can
detect schema changes for autogenerate.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

# Load our app's configuration
from app.core.config import get_settings
from app.core.database import Base

# IMPORTANT: Import ALL models here so Alembic can detect them
from app.models.knowledge import KnowledgeDocument
from app.models.conversation import Conversation, Message

settings = get_settings()

# Alembic Config object (gives access to alembic.ini values)
config = context.config

# Override the sqlalchemy.url from alembic.ini with our real DB URL
# This ensures we always use the URL from .env, not the hardcoded ini value
config.set_main_option("sqlalchemy.url", settings.sync_database_url)

# Configure Python logging from alembic.ini's [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The metadata object that autogenerate compares against
# Add metadata from all models here
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Used when you want to generate SQL scripts without connecting to a DB.
    Call with: alembic upgrade head --sql
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,   # Detect column type changes
        compare_server_default=True,  # Detect server-side default changes
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Execute migrations within an active connection/transaction."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations using an async engine.
    We create a fresh sync connection for the migration transaction.
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # No pooling during migrations
        url=settings.database_url,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations with an active database connection (normal mode)."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
