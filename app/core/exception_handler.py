from fastapi import FastAPI, HTTPException, Request, status

from app.core.exceptions import AlreadyExistError, NotFoundError


def include_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(NotFoundError, not_found_exception_handler)
    app.add_exception_handler(AlreadyExistError, conflict_exception_handler)


def not_found_exception_handler(request: Request, exc: NotFoundError) -> None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc) or 'Not Found') from exc


def conflict_exception_handler(request: Request, exc: AlreadyExistError) -> None:
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc) or 'Conflict') from exc
