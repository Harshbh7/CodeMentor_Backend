"""
CodeMentor AI - Shared Pydantic Response Schemas
=================================================
Defines standardized API response envelopes used across all endpoints.

Design Rationale:
- Consistent response format lets frontend/clients write generic handlers.
- `success`, `message`, and `data` are always present — predictable contract.
- Pagination metadata is built-in for list endpoints.
"""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

# Generic type for the data payload
DataT = TypeVar("DataT")


class BaseResponse(BaseModel):
    """
    Standard API response envelope.

    All API responses should use this structure:
    {
        "success": true,
        "message": "Operation completed successfully.",
        "data": { ... }
    }
    """

    success: bool = Field(description="Whether the operation succeeded")
    message: str = Field(description="Human-readable status message")
    data: Any = Field(default=None, description="Response payload")


class SuccessResponse(BaseResponse, Generic[DataT]):
    """
    Generic success response with typed data payload.

    Usage:
        return SuccessResponse[UserResponse](
            success=True,
            message="User fetched.",
            data=user_data,
        )
    """

    success: bool = True
    data: DataT | None = None


class ErrorResponse(BaseModel):
    """
    Standardized error response body.

    Returned by FastAPI exception handlers for all error cases.
    """

    success: bool = False
    message: str = Field(description="Error description")
    error_code: str | None = Field(default=None, description="Machine-readable error code")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (validation field errors, etc.)",
    )


class PaginatedResponse(BaseModel, Generic[DataT]):
    """
    Paginated list response envelope.

    Usage:
        return PaginatedResponse[UserResponse](
            items=users,
            total=100,
            page=1,
            page_size=20,
        )
    """

    success: bool = True
    items: list[DataT]
    total: int = Field(description="Total number of records matching the query")
    page: int = Field(description="Current page number (1-indexed)")
    page_size: int = Field(description="Number of items per page")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there is a next page")
    has_previous: bool = Field(description="Whether there is a previous page")

    @classmethod
    def create(
        cls,
        items: list[DataT],
        total: int,
        page: int,
        page_size: int,
    ) -> "PaginatedResponse[DataT]":
        """
        Factory method to calculate pagination metadata automatically.

        Args:
            items:     The list of items for the current page.
            total:     Total number of matching records.
            page:      Current page number (1-indexed).
            page_size: Items per page.
        """
        import math
        total_pages = math.ceil(total / page_size) if page_size > 0 else 0
        return cls(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1,
        )


class HealthCheckResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str = Field(description="Overall health status: 'healthy' or 'degraded'")
    version: str = Field(description="Application version")
    environment: str = Field(description="Runtime environment")
    services: dict[str, str] = Field(
        description="Individual service health statuses",
        examples=[{"database": "healthy", "redis": "healthy"}],
    )
