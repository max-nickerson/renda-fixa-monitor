"""Optional: TradingView's official MCP server as a data source (eurobond prices, news).

Requires an Essential+ TradingView plan and `pip install "mcp>=1.10"`.
Run `python -m rfmonitor tv-login` once to sign in; tokens are stored in data/tradingview_oauth.json.
Market data from this server is delayed and the server is in beta — treat it as best-effort.
"""
from __future__ import annotations

import asyncio
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from ..config import DATA_DIR, USER_AGENT

SERVER_URL = "https://mcp.tradingview.com/mcp"
CALLBACK_PORT = 3119
TOKEN_FILE = DATA_DIR / "tradingview_oauth.json"


def available() -> bool:
    try:
        import mcp  # noqa: F401
        return True
    except ImportError:
        return False


class _FileStorage:
    """Persists OAuth tokens + dynamic client registration between runs."""

    def _load(self) -> dict:
        return json.loads(TOKEN_FILE.read_text()) if TOKEN_FILE.exists() else {}

    def _save(self, key: str, value) -> None:
        d = self._load()
        d[key] = value.model_dump(mode="json")
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(json.dumps(d))

    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken
        d = self._load().get("tokens")
        return OAuthToken.model_validate(d) if d else None

    async def set_tokens(self, tokens) -> None:
        self._save("tokens", tokens)

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull
        d = self._load().get("client")
        return OAuthClientInformationFull.model_validate(d) if d else None

    async def set_client_info(self, info) -> None:
        self._save("client", info)


def _wait_for_callback() -> tuple[str, str | None]:
    result: dict = {}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            q = parse_qs(urlparse(self.path).query)
            result["code"] = (q.get("code") or [None])[0]
            result["state"] = (q.get("state") or [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write("<h3>TradingView conectado. Pode fechar esta aba.</h3>".encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", CALLBACK_PORT), H)
    t = threading.Thread(target=srv.handle_request, daemon=True)
    t.start()
    t.join(timeout=300)
    srv.server_close()
    if not result.get("code"):
        raise RuntimeError("TradingView sign-in timed out")
    return result["code"], result.get("state")


def _auth(interactive: bool):
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    async def redirect_handler(url: str) -> None:
        if not interactive:
            raise RuntimeError("TradingView session expired — run `python -m rfmonitor tv-login`")
        print(f"Opening browser for TradingView sign-in:\n{url}")
        webbrowser.open(url)

    async def callback_handler() -> tuple[str, str | None]:
        return await asyncio.to_thread(_wait_for_callback)

    return OAuthClientProvider(
        server_url=SERVER_URL,
        client_metadata=OAuthClientMetadata(
            client_name="renda-fixa-monitor",
            redirect_uris=[f"http://localhost:{CALLBACK_PORT}/callback"],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
        ),
        storage=_FileStorage(),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


async def _call(tool_hint: str, args: dict, interactive: bool = False):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(SERVER_URL, auth=_auth(interactive),
                                     headers={"User-Agent": USER_AGENT}) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            name = next((t.name for t in tools if tool_hint in t.name.replace("-", "_")), None)
            if name is None:
                raise RuntimeError(f"TradingView MCP has no tool matching {tool_hint!r}")
            res = await session.call_tool(name, args)
            text = "".join(getattr(c, "text", "") for c in res.content)
            return json.loads(text) if text.strip().startswith(("{", "[")) else text


def call(tool_hint: str, args: dict, interactive: bool = False):
    return asyncio.run(_call(tool_hint, args, interactive))


def login() -> None:
    res = call("search_symbols", {"query": "PETR4"}, interactive=True)
    print("Connected to TradingView MCP.", str(res)[:200])


def search(query: str) -> list[dict]:
    res = call("search_symbols", {"query": query})
    return (res.get("data") or {}).get("symbols", []) if isinstance(res, dict) else []


def ohlcv(symbol: str, interval: str = "1D", count: int = 300) -> list[dict]:
    res = call("get_ohlcv", {"symbol": symbol, "interval": interval, "count": count})
    return res.get("bars", []) if isinstance(res, dict) else []
