"""Run authenticated ModelBest and OpenAI Realtime compatibility APIs.

Run from the repository root: python runtime/realtime.py --help
The existing Gateway and Worker must already be running.
"""
import argparse
import asyncio
from dataclasses import dataclass, field
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import secrets
import ssl
import time
from urllib.parse import urlparse

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
import uvicorn

from gateway_modules.realtime import ProtocolError, RealtimeSession
from gateway_modules.modelbest_realtime import MODEL_ALIASES, ModelBestSession

ROOT = Path(__file__).resolve().parents[1]
OPENAI_MODELS = ("duplex", "gpt-realtime", "gpt-4o-realtime-preview")
MODELS = (*MODEL_ALIASES, *OPENAI_MODELS)


@dataclass
class Settings:
    api_key: str = field(repr=False)
    gateway_url: str = "ws://127.0.0.1:8040"
    upstream_ssl: ssl.SSLContext | None = None
    allowed_origins: tuple[str, ...] = ()
    max_connections: int = 8
    max_session_seconds: int = 1800
    max_buffer_seconds: int = 30
    prepare_timeout: int = 120

    def __post_init__(self):
        if len(self.api_key) < 32 or not self.api_key.isascii() or any(c.isspace() for c in self.api_key):
            raise ValueError("The API key must contain at least 32 non-whitespace ASCII characters")
        parsed = urlparse(self.gateway_url)
        if parsed.scheme not in ("ws", "wss") or not parsed.hostname or parsed.path not in ("", "/") or parsed.query or parsed.username:
            raise ValueError("gateway-url must be a ws:// or wss:// base URL without credentials or a path")
        if parsed.scheme == "wss" and self.upstream_ssl is None:
            self.upstream_ssl = ssl.create_default_context()


def api_error(message, code, status=400):
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error", "code": code, "param": None}}, status_code=status)


def bearer(headers):
    value = headers.get("authorization", "")
    scheme, _, token = value.partition(" ")
    return token if scheme.lower() == "bearer" else ""


def create_app(settings):
    app = FastAPI(title="duplex.cpp Realtime API", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings
    app.state.active_connections = 0
    ephemeral = {}

    def master(token):
        return bool(token) and secrets.compare_digest(token.encode(), settings.api_key.encode())

    def prune():
        for digest, entry in list(ephemeral.items()):
            if entry["expires_at"] <= time.time():
                del ephemeral[digest]

    def authorize(token):
        if master(token):
            return True, None
        prune()
        entry = ephemeral.get(hashlib.sha256(token.encode()).hexdigest()) if token else None
        return (True, entry["session"]) if entry else (False, None)

    @app.get("/health")
    async def health():
        # Liveness only; doesn't disclose the upstream address, config or models.
        return {"status": "healthy", "api": "realtime"}

    @app.get("/v1/models")
    async def models(request: Request):
        if not master(bearer(request.headers)):
            return api_error("Invalid API key", "invalid_api_key", 401)
        return {"object": "list", "data": [{"id": model, "object": "model", "created": 0, "owned_by": "duplex.cpp"} for model in MODELS]}

    async def mint(request, beta=False):
        if not master(bearer(request.headers)):
            return api_error("Invalid API key", "invalid_api_key", 401)
        prune()
        if len(ephemeral) >= 256:
            return api_error("Too many unexpired client secrets", "rate_limit_exceeded", 429)
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > 32768:
                return api_error("Request body too large", "request_too_large", 413)
            body.extend(chunk)
        try:
            data = json.loads(body or b"{}")
            if not isinstance(data, dict):
                raise ProtocolError("Request body must be an object")
            if beta:
                initial = data
                ttl = 60
            else:
                if data.keys() - {"session", "expires_after"}:
                    raise ProtocolError("Unsupported client secret option", "unsupported_feature")
                initial = data.get("session", {})
                expires = data.get("expires_after", {"anchor": "created_at", "seconds": 60})
                if not isinstance(expires, dict) or expires.keys() - {"anchor", "seconds"} or expires.get("anchor", "created_at") != "created_at":
                    raise ProtocolError("Use expires_after.anchor created_at")
                ttl = expires.get("seconds", 60)
                if isinstance(ttl, bool) or not isinstance(ttl, int) or not 10 <= ttl <= 600:
                    raise ProtocolError("expires_after.seconds must be between 10 and 600")
            if not isinstance(initial, dict):
                raise ProtocolError("session must be an object")
            model = initial.get("model", "gpt-realtime")
            if model not in OPENAI_MODELS:
                raise ProtocolError("Unsupported model alias", "model_not_found", "model")
            session = RealtimeSession(None, settings, model=model, beta=beta, initial=initial)
        except (ValueError, ProtocolError) as exc:
            return api_error(str(exc), getattr(exc, "code", "invalid_json"))
        token = "ek_" + secrets.token_urlsafe(32)
        expires_at = int(time.time()) + ttl
        ephemeral[hashlib.sha256(token.encode()).hexdigest()] = {"expires_at": expires_at, "session": initial}
        response = ({**session.description(), "client_secret": {"value": token, "expires_at": expires_at}} if beta
                    else {"value": token, "expires_at": expires_at, "session": session.description()})
        return JSONResponse(response, headers={"Cache-Control": "no-store"})

    @app.post("/v1/realtime/client_secrets")
    async def client_secrets(request: Request):
        return await mint(request)

    @app.post("/v1/realtime/sessions")
    async def sessions(request: Request):
        return await mint(request, beta=True)

    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket):
        # Reject before upgrading; never accept API keys in URLs or access logs.
        origin = ws.headers.get("origin")
        if origin and settings.allowed_origins and origin not in settings.allowed_origins:
            await ws.send_denial_response(api_error("Origin is not allowed", "origin_not_allowed", 403))
            return
        protocols = [p.strip() for p in ws.headers.get("sec-websocket-protocol", "").split(",")]
        token = bearer(ws.headers)
        if not token:
            for prefix in ("map-api-key.", "openai-insecure-api-key."):
                token = next((p[len(prefix):] for p in protocols if p.startswith(prefix)), "")
                if token:
                    break
        valid, initial = authorize(token)
        if not valid:
            await ws.send_denial_response(api_error("Invalid or expired API key", "invalid_api_key", 401))
            return
        if ws.query_params.keys() - {"model", "mode", "playground_experience", "protocol"}:
            await ws.send_denial_response(api_error("Unsupported query parameter (put keys in Authorization)", "invalid_query"))
            return
        beta = "realtime=v1" in ws.headers.get("openai-beta", "") or "openai-beta.realtime-v1" in protocols
        model = ws.query_params.get("model", initial.get("model", "gpt-realtime") if initial is not None else None)
        modelbest = "map.realtime" in protocols or model in MODEL_ALIASES or (model is None and not beta and "realtime" not in protocols)
        protocol = ws.query_params.get("protocol", "modelbest" if modelbest else "openai")
        if protocol not in (("modelbest", "qwen-audio-agent") if modelbest else ("openai",)):
            await ws.send_denial_response(api_error("Unsupported protocol for the selected model", "unsupported_protocol"))
            return
        if modelbest and initial is not None:
            await ws.send_denial_response(api_error("This client secret was issued for an OpenAI session", "invalid_api_key", 401))
            return
        if modelbest and not model:
            await ws.send_denial_response(api_error("ModelBest requires the model query parameter", "model_required"))
            return
        model = model or "gpt-realtime"
        if model not in MODELS:
            await ws.send_denial_response(api_error("Unsupported model alias", "model_not_found", 404))
            return
        if modelbest and model not in MODEL_ALIASES:
            await ws.send_denial_response(api_error("Select MiniCPM-o-4.5-Realtime for the ModelBest protocol", "model_not_found", 404))
            return
        mode = ws.query_params.get("mode", "video" if modelbest else "audio")
        if mode not in (("audio", "video") if modelbest else ("audio",)):
            await ws.send_denial_response(api_error("Supported modes: audio/video (ModelBest), audio (OpenAI)", "unsupported_mode"))
            return
        if app.state.active_connections >= settings.max_connections:
            await ws.send_denial_response(api_error("Connection limit reached", "rate_limit_exceeded", 429))
            return
        try:
            session = (ModelBestSession(ws, settings, model=model, mode=mode, protocol=protocol) if modelbest
                       else RealtimeSession(ws, settings, model=model, beta=beta, initial=initial))
        except ProtocolError as exc:
            await ws.send_denial_response(api_error(str(exc), exc.code))
            return
        # Reserve before accept (an await), so simultaneous handshakes can't race.
        app.state.active_connections += 1
        try:
            selected_protocol = "map.realtime" if modelbest and "map.realtime" in protocols else "realtime" if "realtime" in protocols else None
            await ws.accept(subprotocol=selected_protocol)
            await session.run()
        finally:
            app.state.active_connections -= 1

    return app


def load_key(path):
    value = os.environ.get("MODELBEST_API_KEY") or os.environ.get("DUPLEX_REALTIME_API_KEY")
    if value:
        return value
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            os.chmod(path, 0o600)
            stream.write("sk-duplex-" + secrets.token_urlsafe(36) + "\n")
    except FileExistsError:
        pass
    return path.read_text(encoding="utf-8").strip()


def main():
    from config import get_config
    cfg = get_config()
    service = cfg.service
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--gateway-url", help="Existing Gateway ws(s)://host:port; defaults to runtime/config.json")
    parser.add_argument("--gateway-ca", type=Path, help="CA used to verify the Gateway TLS certificate")
    parser.add_argument("--api-key-file", type=Path, default=ROOT / ".local/realtime/api-key")
    parser.add_argument("--ssl-certfile", type=Path)
    parser.add_argument("--ssl-keyfile", type=Path)
    parser.add_argument("--allowed-origins", default="", help="Comma-separated browser origins")
    parser.add_argument("--max-connections", type=int, default=8)
    parser.add_argument("--http", action="store_true", help="Use HTTP on loopback behind a TLS reverse proxy")
    args = parser.parse_args()
    if args.max_connections < 1 or not 1 <= args.port <= 65535:
        parser.error("Invalid connection limit or port")
    gateway_host = service.gateway_host
    if gateway_host in ("0.0.0.0", "::"):
        gateway_host = "127.0.0.1"
    if ":" in gateway_host:
        gateway_host = f"[{gateway_host}]"
    gateway_url = args.gateway_url or f"{'wss' if service.gateway_https else 'ws'}://{gateway_host}:{service.gateway_port}"
    context = None
    if gateway_url.startswith("wss:"):
        ca = args.gateway_ca
        if ca is None and args.gateway_url is None:
            local_ca = ROOT / "runtime/certs/lan-ca.crt"
            if local_ca.exists():
                ca = local_ca
        context = ssl.create_default_context(cafile=str(ca) if ca else None)
    tls = {}
    if not args.http:
        cert = args.ssl_certfile or (ROOT / "runtime" / service.gateway_ssl_certfile if service.gateway_https else None)
        key = args.ssl_keyfile or (ROOT / "runtime" / service.gateway_ssl_keyfile if service.gateway_https else None)
        if cert and key:
            if not cert.is_file() or not key.is_file():
                parser.error("TLS certificate/key files are missing")
            tls = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
    try:
        loopback = args.host == "localhost" or ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        loopback = False
    if not loopback and not tls:
        parser.error("A LAN/public bind requires TLS certificate/key files; use loopback --http behind a TLS proxy")
    settings = Settings(api_key=load_key(args.api_key_file), gateway_url=gateway_url, upstream_ssl=context,
                        allowed_origins=tuple(o.strip() for o in args.allowed_origins.split(",") if o.strip()),
                        max_connections=args.max_connections)
    print(f"Realtime: {'wss' if tls else 'ws'}://{args.host}:{args.port}/v1/realtime?mode=audio&model=MiniCPM-o-4.5-Realtime", flush=True)
    print(f"API key: MODELBEST_API_KEY / DUPLEX_REALTIME_API_KEY or local file {args.api_key_file} (never put it in a URL)", flush=True)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, access_log=False,
                log_level="warning",
                ws="websockets",
                ws_max_size=8 * 1024 * 1024, ws_max_queue=8,
                timeout_keep_alive=10, **tls)


if __name__ == "__main__":
    main()
