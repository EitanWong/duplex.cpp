#!/usr/bin/env node
// Check a model server from the qwen application's computer, without a WebUI.
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { parseArgs, parseEnv } from 'node:util'
import { fileURLToPath } from 'node:url'

const { values } = parseArgs({ options: {
  config: { type: 'string', default: fileURLToPath(new URL('./config.env', import.meta.url)) },
} })
const settings = parseEnv(await readFile(values.config, 'utf8'))
assert.equal(settings.QWEN_AUDIO_REALTIME_PROVIDER, 'minicpm-o', 'Select the MiniCPM-o Provider')
const endpoint = new URL(settings.QWEN_AUDIO_REALTIME_URL)
assert.equal(endpoint.protocol, 'wss:', 'Use the authenticated TLS model endpoint')
assert(!endpoint.username && !endpoint.password && !endpoint.searchParams.has('api_key'), 'Keep credentials out of URLs')
assert.equal(endpoint.searchParams.get('protocol'), 'qwen-audio-agent', 'Missing qwen wire protocol option')
const key = settings.QWEN_AUDIO_REALTIME_API_KEY
assert(/^[A-Za-z0-9_.-]{32,}$/.test(key || ''), 'Missing local API key')
const modelsUrl = new URL(endpoint)
modelsUrl.protocol = 'https:'
modelsUrl.pathname = '/v1/models'
modelsUrl.search = ''
const models = await fetch(modelsUrl, {
  headers: { Authorization: `Bearer ${key}` }, redirect: 'error', signal: AbortSignal.timeout(10000),
})
assert.equal(models.status, 200, 'Model server authentication failed')
const model = endpoint.searchParams.get('model')
assert((await models.json()).data.some(item => item.id === model), 'Model not advertised by this server')
const unauthorized = await fetch(modelsUrl, {
  headers: { Authorization: 'Bearer invalid-client-check-key' }, redirect: 'error', signal: AbortSignal.timeout(10000),
})
assert.equal(unauthorized.status, 401, 'Server must reject an invalid key')

async function session() {
  const started = performance.now()
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(endpoint, ['map.realtime', `map-api-key.${key}`])
    let initialized = false, closed = false, failure
    const timer = setTimeout(() => {
      failure = new Error('Session initialization/close timed out; check worker readiness or queue')
      ws.close()
      reject(failure)
    }, 120000)
    const fail = error => {
      failure = error
      clearTimeout(timer)
      ws.close()
      reject(error)
    }
    ws.addEventListener('message', message => {
      try {
        const event = JSON.parse(message.data)
        if (event.type === 'session.queue_done') {
          ws.send(JSON.stringify({ type: 'session.init', payload: { system_prompt: 'You are a helpful voice assistant.' } }))
        } else if (event.type === 'session.created') {
          assert.equal(event.model, model)
          initialized = true
          ws.send(JSON.stringify({ type: 'session.close', reason: 'client_connection_check' }))
        } else if (event.type === 'session.closed') {
          closed = true
        } else if (event.type === 'error') {
          throw new Error(event.error?.message || 'Model session failed')
        }
      } catch (error) { fail(error) }
    })
    ws.addEventListener('error', () => fail(new Error('WSS connection failed; check the address, CA trust and API key')))
    ws.addEventListener('close', event => {
      clearTimeout(timer)
      if (failure) return
      if (!initialized || !closed || event.code !== 1000) {
        reject(new Error(`Session did not initialize and close cleanly (code ${event.code})`))
      } else resolve({ initialized, clean_close: closed, elapsed_s: (performance.now() - started) / 1000 })
    })
  })
}

const first = await session()
const reconnect = await session()
console.log(JSON.stringify({
  endpoint: endpoint.href, tls_verified: true, authenticated: true,
  invalid_key_rejected: true, first, reconnect,
}, null, 2))
