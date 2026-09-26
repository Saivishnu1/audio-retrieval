"""Postgres connection helper. Uses psycopg2 (see GUIDELINES.md troubleshooting)."""

from __future__ import annotations

import os
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def build_db_url(
    user: str | None = None,
    password: str | None = None,
    host: str | None = None,
    port: str | None = None,
    dbname: str | None = None,
) -> str:
    """Build a postgresql+psycopg2 URL from args or POSTGRES_* env vars."""
    user = user or os.environ["POSTGRES_USER"]
    password = password or os.environ["POSTGRES_PASSWORD"]
    host = host or os.environ["POSTGRES_HOST"]
    port = port or os.environ["POSTGRES_PORT"]
    dbname = dbname or os.environ["POSTGRES_DB"]
    return (
        f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{dbname}"
    )


_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(build_db_url(), pool_pre_ping=True)
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()
