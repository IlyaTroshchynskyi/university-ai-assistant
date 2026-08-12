from dataclasses import dataclass
from typing import Callable

from fastapi import FastAPI
from starlette.routing import Mount


@dataclass(frozen=True, kw_only=True, slots=True)
class _DepOverride:
    dependency: Callable
    override: Callable


def override_app_test_dependencies(app: FastAPI) -> None:
    deps: list[_DepOverride] = []
    for dep in deps:
        override_dependency(app, dep.dependency, dep.override)


def override_dependency(app: FastAPI, dependency: Callable, override: Callable) -> None:
    app.dependency_overrides[dependency] = override

    for route in app.router.routes:
        if isinstance(route, Mount):
            route.app.dependency_overrides[dependency] = override


def remove_dependency_override(app: FastAPI, dependency: Callable) -> None:
    app.dependency_overrides.pop(dependency, None)

    for route in app.router.routes:
        if isinstance(route, Mount):
            route.app.dependency_overrides.pop(dependency, None)
