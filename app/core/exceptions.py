"""
CodeMentor AI - Custom Exception Hierarchy
==========================================
Centralizes all application-specific exceptions.

Design Rationale:
- Custom exceptions make error handling explicit and traceable.
- Each exception maps to an HTTP status code, making FastAPI exception
  handlers clean and simple.
- Structured error details allow the API to return consistent error bodies.
"""

from typing import Any


class CodeMentorException(Exception):
    """
    Base exception for all CodeMentor AI application errors.
    All custom exceptions should inherit from this class.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


# ----------------------------------------------------------
# Authentication & Authorization
# ----------------------------------------------------------

class AuthenticationError(CodeMentorException):
    """Raised when a user provides invalid credentials."""

    def __init__(self, message: str = "Authentication failed.") -> None:
        super().__init__(message=message, status_code=401)


class AuthorizationError(CodeMentorException):
    """Raised when an authenticated user lacks required permissions."""

    def __init__(self, message: str = "You do not have permission to perform this action.") -> None:
        super().__init__(message=message, status_code=403)


class TokenExpiredError(CodeMentorException):
    """Raised when a JWT token has expired."""

    def __init__(self, message: str = "Token has expired. Please log in again.") -> None:
        super().__init__(message=message, status_code=401)


class InvalidTokenError(CodeMentorException):
    """Raised when a JWT token is malformed or invalid."""

    def __init__(self, message: str = "Invalid authentication token.") -> None:
        super().__init__(message=message, status_code=401)


# ----------------------------------------------------------
# Resource Errors
# ----------------------------------------------------------

class NotFoundError(CodeMentorException):
    """Raised when a requested resource does not exist."""

    def __init__(self, resource: str = "Resource", identifier: Any = None) -> None:
        message = f"{resource} not found."
        if identifier is not None:
            message = f"{resource} with id '{identifier}' not found."
        super().__init__(message=message, status_code=404)


class AlreadyExistsError(CodeMentorException):
    """Raised when attempting to create a resource that already exists."""

    def __init__(self, resource: str = "Resource", field: str = "id", value: Any = None) -> None:
        message = f"{resource} already exists."
        if value is not None:
            message = f"{resource} with {field}='{value}' already exists."
        super().__init__(message=message, status_code=409)


# ----------------------------------------------------------
# Validation
# ----------------------------------------------------------

class ValidationError(CodeMentorException):
    """Raised when input data fails business-logic validation."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, status_code=422, details=details or {})


# ----------------------------------------------------------
# External Service Errors
# ----------------------------------------------------------

class AIServiceError(CodeMentorException):
    """Raised when the Gemini AI service returns an error or times out."""

    def __init__(self, message: str = "AI service is currently unavailable.") -> None:
        super().__init__(message=message, status_code=503)


class VectorDBError(CodeMentorException):
    """Raised when ChromaDB operations fail."""

    def __init__(self, message: str = "Vector database operation failed.") -> None:
        super().__init__(message=message, status_code=503)


class DatabaseError(CodeMentorException):
    """Raised when a PostgreSQL database operation fails unexpectedly."""

    def __init__(self, message: str = "A database error occurred.") -> None:
        super().__init__(message=message, status_code=500)


# ----------------------------------------------------------
# File / Upload Errors
# ----------------------------------------------------------

class FileTooLargeError(CodeMentorException):
    """Raised when an uploaded file exceeds the maximum allowed size."""

    def __init__(self, max_mb: int = 50) -> None:
        super().__init__(
            message=f"File exceeds the maximum allowed size of {max_mb}MB.",
            status_code=413,
        )


class UnsupportedFileTypeError(CodeMentorException):
    """Raised when an unsupported file format is uploaded."""

    def __init__(self, file_type: str, allowed: list[str] | None = None) -> None:
        allowed_str = ", ".join(allowed) if allowed else "PDF, TXT"
        super().__init__(
            message=f"File type '{file_type}' is not supported. Allowed: {allowed_str}.",
            status_code=415,
        )


# ----------------------------------------------------------
# Rate Limiting
# ----------------------------------------------------------

class RateLimitExceededError(CodeMentorException):
    """Raised when a user exceeds the request rate limit."""

    def __init__(self, message: str = "Rate limit exceeded. Please try again later.") -> None:
        super().__init__(message=message, status_code=429)
