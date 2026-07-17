"""
CodeMentor AI - Base ORM Model
================================
Defines a shared abstract base class with common columns that all
database models should include.

Design Rationale:
- Centralizing `id`, `created_at`, `updated_at` in a mixin avoids repetition
  and guarantees consistency across all tables.
- Using UUID as the primary key (instead of integer sequences) is safer
  for distributed systems and prevents enumeration attacks on APIs.
- `updated_at` with `onupdate` ensures the timestamp is auto-maintained
  by SQLAlchemy, not the application code.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class TimestampMixin:
    """
    Mixin that adds `created_at` and `updated_at` timestamp columns.
    Inherit this in any model that needs audit timestamps.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),  # Set by PostgreSQL on INSERT
        nullable=False,
        comment="Record creation timestamp (UTC)",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),        # Automatically updated by PostgreSQL on UPDATE
        nullable=False,
        comment="Record last-update timestamp (UTC)",
    )


class UUIDMixin:
    """
    Mixin that adds a UUID primary key column.
    Using UUID v4 for global uniqueness and security (no enumeration).
    """

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,         # Generate UUID in Python (not DB) for consistency
        nullable=False,
        comment="Globally unique record identifier",
    )


class BaseModel(UUIDMixin, TimestampMixin, Base):
    """
    Abstract base model combining UUID primary key + timestamps.

    All ORM models should inherit from this unless they have a special
    reason to use a different key type.

    Usage:
        from app.models.base import BaseModel

        class User(BaseModel):
            __tablename__ = "users"
            email: Mapped[str] = mapped_column(String(255), unique=True)
    """

    __abstract__ = True  # SQLAlchemy will NOT create a table for this class

    def to_dict(self) -> dict:
        """
        Serialize the model to a plain Python dict.
        Useful for logging and debugging.
        """
        return {
            col.name: getattr(self, col.name)
            for col in self.__table__.columns
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id}>"
