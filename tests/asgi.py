"""A tiny synchronous client over httpx's ASGI transport.

Starlette's ``TestClient`` is pinned to older httpx APIs; driving the app
through ``httpx.ASGITransport`` works with every httpx >= 0.23.
"""

from __future__ import annotations

import asyncio

import httpx


class SyncASGIClient:
    def __init__(self, app) -> None:
        self.app = app

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def go() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(go())

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)
