param([switch]$Video)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$service = (Get-Content -LiteralPath (Join-Path $projectRoot 'runtime/config.json') -Raw | ConvertFrom-Json).service
if (-not $service.gateway_https) { throw 'Enable the LAN TLS service before exporting a remote client configuration.' }
$apiHost = [string]$service.gateway_host
if ($apiHost -in @('', 'localhost', '127.0.0.1', '::1', '0.0.0.0', '::')) { throw 'Set gateway_host to the model host LAN address first.' }
if ($apiHost.Contains(':')) { $apiHost = "[$apiHost]" }
$mode = if ($Video) { 'video' } else { 'audio' }
$apiUrl = "wss://${apiHost}:8006/v1/realtime?mode=$mode&model=MiniCPM-o-4.5-Realtime&protocol=qwen-audio-agent"
$parsed = [Uri]$apiUrl
if (-not $parsed.Host -or $parsed.UserInfo -or $apiUrl -match '["\r\n]') { throw 'Invalid LAN model address.' }
$apiKey = if ($env:MODELBEST_API_KEY) { $env:MODELBEST_API_KEY } elseif ($env:DUPLEX_REALTIME_API_KEY) { $env:DUPLEX_REALTIME_API_KEY }
          else { (Get-Content -LiteralPath (Join-Path $projectRoot '.local/realtime/api-key') -Raw).Trim() }
if ($apiKey -notmatch '^[A-Za-z0-9_.-]{32,}$') { throw 'Use an API key with at least 32 letters, digits, underscores, dots or hyphens.' }
$caPath = Join-Path $projectRoot 'runtime/certs/lan-ca.crt'
if (-not (Test-Path -LiteralPath $caPath)) { throw 'LAN CA public certificate is missing.' }
$outputRoot = Join-Path $projectRoot '.local/qwen-client'
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$lines = @(
    '# Generated client connection; keep this file private.',
    'QWEN_AUDIO_REALTIME_PROVIDER=minicpm-o',
    "QWEN_AUDIO_REALTIME_URL=`"$apiUrl`"",
    "QWEN_AUDIO_REALTIME_API_KEY=$apiKey",
    'QWEN_AUDIO_REALTIME_MODEL=',
    'QWEN_AUDIO_REALTIME_VOICE=',
    'AGENT_PROTOCOL=none'
)
[IO.File]::WriteAllLines((Join-Path $outputRoot 'config.env'), $lines, [Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath $caPath -Destination (Join-Path $outputRoot 'lan-ca.crt') -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'deploy/qwen-client-setup.md') -Destination (Join-Path $outputRoot 'README.md') -Force
Copy-Item -LiteralPath (Join-Path $projectRoot 'scripts/benchmarks/qwen-client-check.mjs') -Destination (Join-Path $outputRoot 'check-connection.mjs') -Force
$archive = Join-Path $projectRoot '.local/qwen-client.zip'
$files = @('config.env', 'lan-ca.crt', 'README.md', 'check-connection.mjs') | ForEach-Object { Join-Path $outputRoot $_ }
Compress-Archive -LiteralPath $files -DestinationPath $archive -Force
Write-Output "Client configuration: $archive"
Write-Output "Model endpoint: $apiUrl"
Write-Output 'Includes the API key and public CA certificate; no server private keys or model weights.'
