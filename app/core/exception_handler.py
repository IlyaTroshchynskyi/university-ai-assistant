from fastapi import FastAPI, HTTPException, Request, status

from app.core.exceptions import (
    AlreadyExistError,
    ConflictingStatusError,
    DocumentTooLargeError,
    NotFoundError,
    UnsupportedDocumentError,
)


def include_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(NotFoundError, not_found_exception_handler)
    app.add_exception_handler(AlreadyExistError, conflict_exception_handler)
    app.add_exception_handler(ConflictingStatusError, conflicting_status_exception_handler)
    app.add_exception_handler(UnsupportedDocumentError, raise_unsupported_media_type)
    app.add_exception_handler(DocumentTooLargeError, raise_content_too_large)


def not_found_exception_handler(request: Request, exc: NotFoundError) -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc) or 'Not Found') from exc


def conflict_exception_handler(request: Request, exc: AlreadyExistError) -> None:
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc) or 'Conflict') from exc


def conflicting_status_exception_handler(request: Request, exc: ConflictingStatusError) -> None:
    """The resource exists, but not in a state this request makes sense for — a booking decision on
    a thread that is not paused, or one decision for a pause covering two actions. 409 like the
    handler above, and separate from it because "already exists" is a different claim."""
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc) or 'Conflict') from exc


def raise_unsupported_media_type(request: Request, error: UnsupportedDocumentError) -> None:
    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail=str(error) or 'Unsupported Media Type',
    ) from error


def raise_content_too_large(request: Request, error: DocumentTooLargeError) -> None:
    raise HTTPException(
        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        detail=str(error) or 'Content Too Large',
    ) from error
