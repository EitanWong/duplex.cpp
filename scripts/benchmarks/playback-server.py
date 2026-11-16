#!/usr/bin/env python3
"""Local-only WebAudio observer for live replay or an external saved-packet producer.

No microphone, browser permissions or external service. The browser uses the
production AudioPlayer, with an AudioWorklet after mixing to observe rendering.
Reports contain timing/levels, never raw audio; all outputs belong in verification/.
"""
import argparse
import asyncio
import json
import re
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

ROOT = Path(__file__).resolve().parents[2]


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    clients = {}

    app = FastAPI()

    @app.websocket('/stream')
    async def socket(ws: WebSocket):
        await ws.accept()
        observer = ws.query_params.get('role') == 'observer'
        run = ws.query_params.get('run', 'default')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', run):
            await ws.close(code=1008)
            return
        journal = None
        if observer:
            clients[ws] = run
            directory = args.output if run == 'default' else args.output / run
            directory.mkdir(parents=True, exist_ok=True)
            journal = (directory / 'browser-events.jsonl').open('a', encoding='utf-8', buffering=1)
        try:
            while True:
                raw = await ws.receive_text()
                event = json.loads(raw)
                if observer:
                    journal.write(json.dumps(event, ensure_ascii=False) + '\n')
                else:
                    # Native replay already saved its PCM; avoid another copy.
                    for client, client_run in tuple(clients.items()):
                        if client_run != run:
                            continue
                        try:
                            await client.send_text(raw)
                        except (WebSocketDisconnect, RuntimeError):
                            clients.pop(client, None)
        except WebSocketDisconnect:
            pass
        finally:
            clients.pop(ws, None)
            if journal:
                journal.close()
    @app.get('/')
    async def page():
        return FileResponse(Path(__file__).with_name('playback.html'))

    @app.get('/status')
    async def status():
        return {'observers': len(clients)}

    app.mount('/player', StaticFiles(directory=ROOT / 'runtime/static/duplex/lib'))
    print(f'Playback observer: http://127.0.0.1:{args.port}', flush=True)
    await uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=args.port, log_level='warning')).serve()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=19280)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
