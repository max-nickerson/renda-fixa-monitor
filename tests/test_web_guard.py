"""Public-hosting guard: reads are open, writes need the edit key (ADMIN_KEY)."""
from fastapi.testclient import TestClient

from rfmonitor import web
from rfmonitor.config import settings


def _client(monkeypatch, key):
    monkeypatch.setattr(settings, "admin_key", key)
    return TestClient(web.create_app(scheduler=False), follow_redirects=False)


def test_open_when_no_key(monkeypatch):
    c = _client(monkeypatch, "")
    assert c.get("/healthz").status_code == 200
    r = c.post("/remove/XX0000000000")  # allowed through the guard (then handled by the route)
    assert not r.headers.get("location", "").startswith("/unlock")


def test_writes_need_key(monkeypatch):
    c = _client(monkeypatch, "s3cret")
    assert c.get("/healthz").status_code == 200
    r = c.post("/remove/XX0000000000", headers={"referer": "http://testserver/"})
    assert r.status_code == 303 and r.headers["location"].startswith("/unlock")
    bad = c.post("/unlock", data={"key": "nope", "next": "/"})
    assert "error=1" in bad.headers["location"]
    ok = c.post("/unlock", data={"key": "s3cret", "next": "/alerts"})
    assert ok.headers["location"] == "/alerts" and web.EDIT_COOKIE in ok.headers.get("set-cookie", "")
    c.cookies.set(web.EDIT_COOKIE, "s3cret")
    r2 = c.post("/remove/XX0000000000")
    assert not r2.headers.get("location", "").startswith("/unlock")


def test_unlock_blocks_open_redirect(monkeypatch):
    c = _client(monkeypatch, "s3cret")
    r = c.post("/unlock", data={"key": "s3cret", "next": "https://evil.example/"})
    assert r.headers["location"] == "/"
