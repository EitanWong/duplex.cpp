// Preloaded by qwen-audio-agent.ps1; keep the installed npm package untouched.
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const packageRoot = process.env.DUPLEX_QWEN_PACKAGE_ROOT
if (!packageRoot) throw new Error('Missing DUPLEX_QWEN_PACKAGE_ROOT for qwen compatibility preload')
const { version } = JSON.parse(await readFile(resolve(packageRoot, 'package.json'), 'utf8'))

if (version === '2.1.0') {
  const { miniCpmOProvider } = await import(pathToFileURL(resolve(
    packageRoot, 'server/src/voice/providers/minicpm-o.mjs',
  )).href)
  // 2.1.0 declares this profile but omits its catalog. WebUI requires both
  // to agree before showing capabilities. Resolve it on each call so the
  // audio/video transport flags remain consistent with the configured URL.
  if (miniCpmOProvider.modelCatalog === undefined) {
    miniCpmOProvider.modelCatalog = () => Object.freeze([miniCpmOProvider.modelProfile()])
  }
}
