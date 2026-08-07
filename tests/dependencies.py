from dataclasses import dataclass
from typing import Callable

from fastapi import FastAPI
from starlette.routing import Mount

from app.ai_assistant_langchain.agent import create_assistant_agent
from tests.agent_stubs import get_test_agent


@dataclass(frozen=True, kw_only=True, slots=True)
class _DepOverride:
    dependency: Callable
    override: Callable


def override_app_test_dependencies(app: FastAPI):
    deps: list[_DepOverride] = [
        _DepOverride(dependency=create_assistant_agent, override=get_test_agent),
    ]
    for dep in deps:
        override_dependency(app, dep.dependency, dep.override)


def override_dependency(app: FastAPI, dependency: Callable, override: Callable) -> None:
    app.dependency_overrides[dependency] = override

    for route in app.router.routes:
        if isinstance(route, Mount):
            route.app.dependency_overrides[dependency] = override
