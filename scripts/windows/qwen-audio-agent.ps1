param(
    [ValidateSet('start', 'stop', 'status', 'run')][string]$Action = 'start',
    [string]$QwenRoot = '',
    [string]$NodePath = '',
    [ValidateRange(1, 65535)][int]$Port = 3101,
    [string]$ApiUrl = '',
    [switch]$Video
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $QwenRoot) { $QwenRoot = Join-Path $projectRoot '.local/qwen-audio-agent/node_modules/qwen-audio-agent' }
$QwenRoot = [IO.Path]::GetFullPath($QwenRoot)
$server = Join-Path $QwenRoot 'server/src/index.mjs'
$compatibilityPreload = [Uri]::new((Join-Path $PSScriptRoot 'qwen-audio-agent-compat.mjs')).AbsoluteUri
$stateRoot = Join-Path $projectRoot '.local/qwen-audio-agent'
$configRoot = Join-Path $stateRoot 'config'
$statePath = Join-Path $stateRoot 'windows.json'
$logRoot = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Force -Path $stateRoot, $configRoot, $logRoot | Out-Null
$controlLock = $null
$savedEnvironment = @{}

function Read-OwnedProcess($State) {
    if (-not $State) { return $null }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$State.processId)"
    if ($process -and $process.CreationDate.ToUniversalTime().Ticks.ToString() -eq [string]$State.created -and
        ([string]$process.CommandLine).Contains([string]$State.server)) { return $process }
    return $null
}

try {
    try { $controlLock = [IO.File]::Open((Join-Path $stateRoot 'windows.lock'), 'OpenOrCreate', 'ReadWrite', 'None') }
    catch { throw 'Another qwen-audio-agent control command is running.' }
    $state = if (Test-Path -LiteralPath $statePath) { Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json } else { $null }
    $owned = Read-OwnedProcess $state
    if ($Action -eq 'status') {
        if ($owned) { Write-Output "qwen-audio-agent running: PID $($owned.ProcessId), $($state.url)" }
        else { Write-Output 'qwen-audio-agent stopped' }
        Write-Output "Isolated config: $(Join-Path $configRoot 'config.env')"
        return
    }
    if ($Action -eq 'stop') {
        # This launcher starts frontend-only mode, with no backend child Agent.
        if ($owned) { Stop-Process -Id $owned.ProcessId -Force -ErrorAction SilentlyContinue }
        Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
        Write-Output 'qwen-audio-agent stopped; model API continues running'
        return
    }
    if (-not (Test-Path -LiteralPath $server)) {
        throw 'Install qwen-audio-agent locally first, or pass -QwenRoot. See docs/qwen-audio-agent.md.'
    }
    if (-not $NodePath) { $NodePath = (Get-Command node -ErrorAction Stop).Source }
    $nodeVersion = & $NodePath -p 'process.versions.node'
    if ($LASTEXITCODE -ne 0) { throw 'Could not determine Node.js version.' }
    $version = [version]$nodeVersion
    if (-not (($version.Major -eq 22 -and $version -ge [version]'22.22.2') -or
              ($version.Major -eq 24 -and $version -ge [version]'24.15.0') -or $version.Major -ge 26)) {
        throw 'qwen-audio-agent requires Node.js ^22.22.2, ^24.15.0 or >=26.0.0.'
    }
    $service = (Get-Content -LiteralPath (Join-Path $projectRoot 'runtime/config.json') -Raw | ConvertFrom-Json).service
    if (-not $ApiUrl) {
        $scheme = if ($service.gateway_https) { 'wss' } else { 'ws' }
        $apiHost = [string]$service.gateway_host
        if ($apiHost -in @('0.0.0.0', '::')) { $apiHost = '127.0.0.1' }
        if ($apiHost.Contains(':')) { $apiHost = "[$apiHost]" }
        $mode = if ($Video) { 'video' } else { 'audio' }
        $ApiUrl = "${scheme}://${apiHost}:8006/v1/realtime?mode=$mode&model=MiniCPM-o-4.5-Realtime&protocol=qwen-audio-agent"
    }
    $parsed = [Uri]$ApiUrl
    if ($parsed.Scheme -notin @('ws', 'wss') -or $parsed.UserInfo -or $ApiUrl.Contains('"') -or $ApiUrl.Contains("`n") -or $ApiUrl.Contains("`r")) {
        throw 'ApiUrl must be a ws/wss URL without credentials or line breaks.'
    }
    if ($owned) {
        if ($state.port -ne $Port -or $state.apiUrl -ne $ApiUrl -or $state.server -ne $server) {
            throw 'qwen-audio-agent is running with different options. Stop it before changing configuration.'
        }
        Write-Output "qwen-audio-agent already running: $($state.url)"
        return
    }
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw "Port $Port is already in use." }
    $apiKey = if ($env:MODELBEST_API_KEY) { $env:MODELBEST_API_KEY } elseif ($env:DUPLEX_REALTIME_API_KEY) { $env:DUPLEX_REALTIME_API_KEY }
              else { (Get-Content -LiteralPath (Join-Path $projectRoot '.local/realtime/api-key') -Raw).Trim() }
    if ($apiKey -notmatch '^[A-Za-z0-9_.-]{32,}$') { throw 'Use a local API key containing at least 32 letters, digits, underscores, dots or hyphens.' }
    $settings = @{
        DUPLEX_QWEN_PACKAGE_ROOT = $QwenRoot
        QWAUDIO_CONFIG_DIR = $configRoot
        QWAUDIO_DATA_DIR = Join-Path $configRoot 'data'
        QWAUDIO_STATE_DIR = Join-Path $configRoot 'state'
        QWAUDIO_CACHE_DIR = Join-Path $configRoot 'cache'
        QWAUDIO_WORKSPACE = Join-Path $configRoot 'data/workspace'
        NODE_EXTRA_CA_CERTS = Join-Path $projectRoot 'runtime/certs/lan-ca.crt'
        QWEN_AUDIO_REALTIME_PROVIDER = 'minicpm-o'
        QWEN_AUDIO_REALTIME_URL = $ApiUrl
        QWEN_AUDIO_REALTIME_API_KEY = $apiKey
        QWEN_AUDIO_REALTIME_MODEL = ''
        QWEN_AUDIO_REALTIME_VOICE = ''
        AGENT_PROTOCOL = 'none'
        HOST = '127.0.0.1'
        PORT = [string]$Port
    }
    if ($parsed.Scheme -eq 'wss' -and -not (Test-Path -LiteralPath $settings.NODE_EXTRA_CA_CERTS)) { throw 'LAN CA certificate is missing.' }
    # Only this project's isolated qwen config is written. Never print the key.
    $lines = @('# Generated duplex.cpp voice frontend configuration')
    foreach ($name in @('QWEN_AUDIO_REALTIME_PROVIDER', 'QWEN_AUDIO_REALTIME_URL', 'QWEN_AUDIO_REALTIME_API_KEY',
                       'QWEN_AUDIO_REALTIME_MODEL', 'QWEN_AUDIO_REALTIME_VOICE', 'AGENT_PROTOCOL', 'HOST', 'PORT')) {
        $lines += "$name=$($settings[$name])"
    }
    [IO.File]::WriteAllLines((Join-Path $configRoot 'config.env'), $lines, [Text.UTF8Encoding]::new($false))
    foreach ($name in $settings.Keys) {
        $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $settings[$name], 'Process')
    }
    if ($Action -eq 'run') {
        & $NodePath --import $compatibilityPreload $server
        if ($LASTEXITCODE -ne 0) { throw "qwen-audio-agent exited with status $LASTEXITCODE" }
        return
    }
    $process = Start-Process -FilePath $NodePath -ArgumentList @('--import', ('"' + $compatibilityPreload + '"'), ('"' + $server + '"')) -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logRoot 'qwen-audio-agent.stdout.log') `
        -RedirectStandardError (Join-Path $logRoot 'qwen-audio-agent.stderr.log')
    $identity = Get-CimInstance Win32_Process -Filter "ProcessId = $($process.Id)"
    if (-not $identity) { throw 'qwen-audio-agent exited; see logs/qwen-audio-agent.stderr.log.' }
    $state = @{ processId = $process.Id; created = $identity.CreationDate.ToUniversalTime().Ticks.ToString()
                server = $server; port = $Port; apiUrl = $ApiUrl; url = "http://127.0.0.1:$Port" }
    $state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
    $ready = $false
    for ($attempt = 0; $attempt -lt 100; $attempt++) {
        if (-not (Read-OwnedProcess $state)) { throw 'qwen-audio-agent exited; see logs/qwen-audio-agent.stderr.log.' }
        try {
            $health = Invoke-RestMethod -Uri "$($state.url)/api/health" -TimeoutSec 1
            $catalogProfile = @($health.realtimeModelCatalog | Where-Object { $_.id -eq $health.realtimeModel })
            if ($health.ok -and $health.realtimeProvider -eq 'minicpm-o' -and $health.voiceConfigured -and $catalogProfile.Count -eq 1) {
                $ready = $true; break
            }
        } catch { }
        Start-Sleep -Milliseconds 300
    }
    if (-not $ready) { throw 'qwen-audio-agent did not become ready; see its logs.' }
    Write-Output "qwen-audio-agent started: $($state.url)"
    Write-Output "Voice API: $ApiUrl"
} finally {
    foreach ($name in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process') }
    if ($controlLock) { $controlLock.Dispose() }
}
