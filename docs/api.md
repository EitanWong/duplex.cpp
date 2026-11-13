# Duplex API quick reference

The public gateway exposes `ws://localhost:8040/ws/duplex/{session_id}`.
Use a unique `adx_`-prefixed ID for audio-only sessions or `omni_` for audio/video.
One session occupies one worker. Listen and send concurrently; do not wait for a
speech response before forwarding the next microphone chunk.

1. Connect and wait for `queue_done`; `queued` / `queue_update` may arrive first.
2. Send `prepare` and wait for `prepared`.
3. Send one second of 16 kHz mono float32 little-endian PCM per `audio_chunk`.
4. Handle `result`, `audio_only` and `interrupted` independently.
5. Send `stop` when finished. `pause` / `resume` are also supported.

```json
{"type":"prepare","system_prompt":"You are a helpful voice assistant.","config":{"generate_audio":true,"chunk_ms":1000}}
```

```json
{"type":"audio_chunk","audio_base64":"<base64 of 16000 float32 PCM samples>"}
```

Optionally add `frame_base64_list: ["<base64 JPEG>"]` to an audio chunk.
Use `force_listen: true` to interrupt speaking and force listening.

| Incoming event | Handling |
| --- | --- |
| `result` | Read `is_listen`, `text`, timing and turn metadata |
| `audio_only` | Decode `audio_data` as raw mono float32 little-endian PCM at 24 kHz |
| `interrupted` | Flush obsolete playback; use `audio_epoch` to discard older output |
| `error` / `timeout` | Stop capture and close or re-establish the session |
| `stopped` | Session released |

Speech is streamed separately from text decisions; `result.audio_data` may be empty.
Honor `speech_end` metadata when present. The native `engine` WebSocket `/backend`
is an internal protocol and uses different messages and raw PCM encoding.

The browser client also supports the existing short-frame endpointing mode.
For a first integration, use the one-second messages above. See `runtime/worker.py`
and `runtime/static/duplex/lib/duplex-session.js` for the current implementation.

## 中文

连接后依次等待 `queue_done`、发送 `prepare`、等待 `prepared`，然后持续发送
1 秒、16 kHz、单声道 float32 小端 PCM 的 Base64 数据（字段为 `audio_base64`）。
收音和播放必须并行。语音经独立 `audio_only` 事件发送，`audio_data` 为
24 kHz 单声道 float32 小端原始 PCM，不含 WAV 文件头，也不保证包含在 `result` 内。
打断时清空旧播放队列，
并利用 `audio_epoch` 隔离过期输出。远程部署使用带认证的 HTTPS/WSS 反向代理。
