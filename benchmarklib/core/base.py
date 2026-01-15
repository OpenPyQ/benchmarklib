from datetime import datetime
from sqlalchemy import Boolean, Column, ForeignKey, Index, Integer, String, JSON, Float, DateTime, LargeBinary, select, create_engine, select, func, or_, text
from sqlalchemy.orm import  declared_attr, Mapped, relationship, mapped_column, sessionmaker, Session, DeclarativeBase, DeclarativeMeta, Mapped, relationship, selectinload, joinedload


class Base(DeclarativeBase):
    """
    Base class for all ORM models, providing common attributes.
    """
    __abstract__ = True

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)