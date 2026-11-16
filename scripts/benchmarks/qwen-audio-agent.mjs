#!/usr/bin/env node
// Exercise the unmodified qwen-audio-agent npm Provider against a live model.
import assert from 'node:assert/strict'
import { readFile, mkdir, writeFile, realpath } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL, fileURLToPath } from 'node:url'
import { parseArgs, parseEnv } from 'node:util'
import { createRequire } from 'node:module'
import { setTimeout as delay } from 'node:timers/promises'

const root = fileURLToPath(new URL('../../', import.meta.url))
const { values } = parseArgs({ options: {
  'qwen-root': { type: 'string', default: resolve(root, '.local/qwen-audio-agent/node_modules/qwen-audio-agent') },
  url: { type: 'string' },
  'key-file': { type: 'string', default: resolve(root, '.local/realtime/api-key') },
  audio: { type: 'string', default: resolve(root, 'runtime/tests/cases/common/user_audio/000_user_audio0.wav') },
  output: { type: 'string', default: resolve(root, 'verification/qwen-audio-agent') },
  seconds: { type: 'string', default: '25' },
  frame: { type: 'string' },
  gateway: { type: 'string' },
  'desktop-settings': { type: 'boolean', default: false },
} })
const seconds = Number(values.seconds)
assert(Number.isInteger(seconds) && seconds >= 15 && seconds <= 120, 'seconds must be 15-120')
if (!values.url) {
  const { service } = JSON.parse(await readFile(resolve(root, 'runtime/config.json'), 'utf8'))
  let host = service.gateway_host
  if (['0.0.0.0', '::'].includes(host)) host = '127.0.0.1'
  if (host.includes(':')) host = `[${host}]`
  values.url = `${service.gateway_https ? 'wss' : 'ws'}://${host}:8006/v1/realtime?mode=${values.frame ? 'video' : 'audio'}&model=MiniCPM-o-4.5-Realtime&protocol=qwen-audio-agent`
}
// Keep the test's generated identity and user state separate from installed apps.
process.env.QWAUDIO_CONFIG_DIR = resolve(root, values['desktop-settings']
  ? '.local/qwen-audio-agent/desktop-verification/config'
  : '.local/qwen-audio-agent/benchmark-config')
for (const [key, directory] of Object.entries({
  DATA_DIR: 'data', STATE_DIR: 'state', CACHE_DIR: 'cache', WORKSPACE: 'data/workspace',
})) process.env[`QWAUDIO_${key}`] = resolve(process.env.QWAUDIO_CONFIG_DIR, directory)
process.env.QWEN_AUDIO_REALTIME_PROVIDER = 'minicpm-o'
process.env.QWEN_AUDIO_REALTIME_URL = values.url
process.env.QWEN_AUDIO_REALTIME_API_KEY = (await readFile(values['key-file'], 'utf8')).trim()
process.env.QWEN_AUDIO_REALTIME_MODEL = ''
process.env.QWEN_AUDIO_REALTIME_VOICE = ''
process.env.AGENT_PROTOCOL = 'none'
const moduleAt = path => import(pathToFileURL(resolve(values['qwen-root'], path)).href)
let desktopSettings = null
if (values['desktop-settings']) {
  assert(!values.gateway, '--desktop-settings verifies the desktop model connection directly; omit --gateway')
  const { createSettingsStore } = await moduleAt('desktop/src/settings-store.mjs')
  const { applySettingsEnvironment, realtimeSettingsConfiguration } = await moduleAt('desktop/src/settings-config.mjs')
  const store = createSettingsStore({
    configDir: process.env.QWAUDIO_CONFIG_DIR,
    clientDir: resolve(process.env.QWAUDIO_CONFIG_DIR, 'desktop-client'),
    env: { ...process.env },
  })
  store.save({
    realtimeProvider: 'minicpm-o', realtimeUrl: values.url,
    realtimeApiKey: process.env.QWEN_AUDIO_REALTIME_API_KEY,
    realtimeModel: '', realtimeVoice: '', agentProtocol: 'none',
  })
  const saved = store.load()
  assert(store.ready(), 'Official desktop settings store rejected this model service')
  assert.equal(saved.realtimeProvider, 'minicpm-o')
  assert.equal(saved.realtimeUrl, values.url, 'Desktop settings changed the model service URL')
  assert.equal(saved.realtimeApiKey, process.env.QWEN_AUDIO_REALTIME_API_KEY)
  assert.equal(saved.realtimeModel, '')
  assert.equal(saved.realtimeVoice, '')
  const runtime = applySettingsEnvironment(saved, { ...process.env, ...parseEnv(await readFile(store.path, 'utf8')) })
  Object.assign(process.env, runtime)
  const connection = realtimeSettingsConfiguration(saved)
  assert(connection.active.configured, 'Desktop runtime rejected the saved connection')
  assert.equal(connection.active.endpoint, values.url)
  assert.equal(connection.credential, saved.realtimeApiKey)
  assert.equal(process.env.QWEN_AUDIO_REALTIME_URL, values.url)
  desktopSettings = { saved_and_reloaded: true, ready: true, runtime_configuration: true }
}
const { RealtimeFrontend } = await moduleAt('server/src/voice/realtime-provider.mjs')
const { miniCpmOProvider } = await moduleAt('server/src/voice/providers/minicpm-o.mjs')
const { version } = JSON.parse(await readFile(resolve(values['qwen-root'], 'package.json'), 'utf8'))

function wavPcm16(wav) {
  assert(wav.toString('ascii', 0, 4) === 'RIFF' && wav.toString('ascii', 8, 12) === 'WAVE', 'Use a RIFF WAV file')
  let pcm
  let validFormat = false
  for (let offset = 12; offset + 8 <= wav.length;) {
    const kind = wav.toString('ascii', offset, offset + 4)
    const size = wav.readUInt32LE(offset + 4)
    const start = offset + 8
    assert(start + size <= wav.length, 'Truncated WAV chunk')
    if (kind === 'fmt ') {
      assert(size >= 16 && wav.readUInt16LE(start) === 1 && wav.readUInt16LE(start + 2) === 1
        && wav.readUInt32LE(start + 4) === 16000 && wav.readUInt16LE(start + 14) === 16,
      'Use 16 kHz mono PCM16 WAV')
      validFormat = true
    }
    if (kind === 'data') pcm = wav.subarray(start, start + size)
    offset = start + size + size % 2
  }
  assert(validFormat && pcm?.length && pcm.length % 2 === 0, 'Missing PCM16 data')
  return pcm
}

function responseWav(pcm) {
  const header = Buffer.alloc(44)
  header.write('RIFF'); header.writeUInt32LE(36 + pcm.length, 4); header.write('WAVEfmt ', 8)
  header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22)
  header.writeUInt32LE(24000, 24); header.writeUInt32LE(48000, 28)
  header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34); header.write('data', 36)
  header.writeUInt32LE(pcm.length, 40)
  return Buffer.concat([header, pcm])
}

const source = wavPcm16(await readFile(values.audio))
const input = Buffer.concat([Buffer.alloc(32000), source])
const frame = values.frame ? (await readFile(values.frame)).toString('base64') : null
const events = [], audio = [], texts = []
const active = new Set()
const responses = new Set(), completed = new Set()
let failure, firstAudio, frontendReady = false
const started = performance.now()
const onEvent = event => {
    const saved = { elapsed_s: (performance.now() - started) / 1000, ...event }
    const responseId = event.responseId || event.response_id || event.response?.id
    if (['response.created', 'response.started'].includes(event.type)) {
      assert(responseId, 'Missing response ID')
      active.add(responseId)
      responses.add(responseId)
    }
    if (event.type === 'response.text.delta') texts.push(event.delta)
    if (event.type === 'transcript.delta' && event.role === 'assistant') texts.push(event.content)
    if (['response.audio.delta', 'audio.delta'].includes(event.type)) {
      assert(active.has(responseId), 'Audio arrived outside an active response')
      const encoded = event.type === 'audio.delta' ? event.audio : event.delta
      if (event.type === 'audio.delta') assert.equal(event.sampleRate, 24000, 'Unexpected output sample rate')
      firstAudio ??= saved.elapsed_s
      audio.push(Buffer.from(encoded, 'base64'))
      saved[event.type === 'audio.delta' ? 'audio' : 'delta'] = `<${encoded.length} Base64 characters>`
    }
    if (['response.done', 'audio.done'].includes(event.type)) {
      assert(responses.has(responseId), 'Completion arrived without a response')
      active.delete(responseId)
      completed.add(responseId)
    }
    if (event.type === 'error') failure = new Error(JSON.stringify(event.error))
    events.push(saved)
}
let frontend
if (values.gateway) {
  const { GatewayClient } = await moduleAt('shared/gateway/client-sdk.mjs')
  const { gatewayReferenceClientCapabilities } = await moduleAt('shared/gateway/client-profiles.mjs')
  const WebSocket = createRequire(await realpath(resolve(values['qwen-root'], 'package.json')))('ws')
  let resolveReady, rejectReady
  const voiceReady = new Promise((resolve, reject) => { resolveReady = resolve; rejectReady = reject })
  const client = new GatewayClient({
    url: values.gateway, createSocket: (url, options) => new WebSocket(url, options),
    clientType: 'web', locale: 'zh-CN', timeZone: 'Asia/Shanghai', reconnect: false,
    capabilities: gatewayReferenceClientCapabilities('web'),
    configure: { voiceEnabled: true, inputEnabled: true, outputEnabled: true, provider: 'minicpm-o' },
    onEvent: event => {
      if (event.type === 'voice.ready') resolveReady()
      if (event.type === 'error') rejectReady(new Error(JSON.stringify(event.error || event)))
      onEvent(event)
    },
    onStatus: status => {
      if (['unavailable', 'occupied', 'revoked', 'replaced'].includes(status.state)) {
        const error = new Error('Gateway client: ' + status.state)
        failure = error; rejectReady(error)
      }
    },
  })
  frontend = {
    ready: false,
    async connect() {
      client.start()
      let timer
      try {
        await Promise.race([voiceReady, new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error('Gateway voice.ready timeout')), 120000)
        })])
        this.ready = true
      } finally { clearTimeout(timer) }
    },
    appendAudio: audio => assert(client.send({ type: 'input_audio_buffer.append', audio })),
    appendImage: image => client.send({ type: 'input_image_buffer.append', image }),
    close: () => client.stop(),
  }
} else frontend = new RealtimeFrontend({
  provider: miniCpmOProvider,
  agentContext: { client: { locale: 'zh-CN' }, backend: { available: false } },
  onEvent,
  onError: error => { failure = error },
})
try {
  await frontend.connect()
  frontendReady = frontend.ready
  assert(frontendReady, 'Official frontend did not become ready')
  const connected = performance.now()
  for (let offset = 0; offset < seconds * 32000; offset += 3200) {
    if (failure) throw failure
    const packet = Buffer.alloc(3200)
    if (offset < input.length) input.copy(packet, 0, offset, Math.min(offset + 3200, input.length))
    if (frame && offset % 32000 === 0) assert(frontend.appendImage(frame), 'Video input was not negotiated')
    frontend.appendAudio(packet.toString('base64'))
    await delay(Math.max(0, connected + (offset + 3200) / 32 - performance.now()))
  }
  if (failure) throw failure
} finally {
  frontend.close()
}
const pcm = Buffer.concat(audio)
const summary = {
  package: `qwen-audio-agent@${version}`, provider: miniCpmOProvider.key,
  ready: frontendReady, input_duration_s: source.length / 32000,
  output_duration_s: pcm.length / 48000, first_audio_s: firstAudio ?? null,
  transcript: texts.join(''), events: events.length,
  responses: responses.size, completed_responses: completed.size,
  active_responses: active.size,
  video: Boolean(frame), elapsed_s: (performance.now() - started) / 1000,
  gateway: values.gateway || null,
  desktop_settings: desktopSettings,
}
await mkdir(values.output, { recursive: true })
await writeFile(resolve(values.output, 'events.json'), JSON.stringify(events, null, 2))
await writeFile(resolve(values.output, 'summary.json'), JSON.stringify(summary, null, 2))
if (pcm.length) await writeFile(resolve(values.output, 'response.wav'), responseWav(pcm))
console.log(JSON.stringify(summary, null, 2))
assert(texts.length && pcm.length, 'Missing text or audio through official qwen Provider')
assert(summary.completed_responses > 0, 'No normalized response completed')
assert.equal(active.size, 0, 'A response was still active at the end; increase --seconds')
