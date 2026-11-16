param(
    [ValidateSet('start', 'stop', 'status', 'run')][string]$Action = 'start',
    [string]$ListenHost = '0.0.0.0',
    [ValidateRange(1, 65535)][int]$Port = 8006,
    [string]$AllowedOrigins = '',
    [switch]$Http
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
$server = Join-Path $projectRoot 'runtime/realtime.py'
$stateRoot = Join-Path $projectRoot '.local/realtime'
$statePath = Join-Path $stateRoot 'windows.json'
$logRoot = Join-Path $projectRoot 'logs'
New-Item -ItemType Directory -Force -Path $stateRoot, $logRoot | Out-Null
$controlLock = $null

function Read-OwnedProcess($State) {
    if (-not $State) { return $null }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$State.processId)"
    if ($process -and $process.CreationDate.ToUniversalTime().Ticks.ToString() -eq [string]$State.created -and
        ([string]$process.CommandLine).Contains($server)) { return $process }
    return $null
}

try {
    try { $controlLock = [IO.File]::Open((Join-Path $stateRoot 'windows.lock'), 'OpenOrCreate', 'ReadWrite', 'None') }
    catch { throw 'Another Realtime start/stop command is running.' }
    $state = if (Test-Path -LiteralPath $statePath) { Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json } else { $null }
    $owned = Read-OwnedProcess $state
    if ($Action -eq 'status') {
        if ($owned) { Write-Output "Realtime running: PID $($owned.ProcessId), $($state.url)" }
        else { Write-Output 'Realtime stopped' }
        Write-Output "Key file: $(Join-Path $stateRoot 'api-key')"
        return
    }
    if ($Action -eq 'stop') {
        if ($owned) {
            # Capture descendants before stopping the venv shim. Validate creation
            # times so reused PIDs cannot make this command stop another service.
            $snapshot = @(Get-CimInstance Win32_Process)
            $targets = @($owned)
            do {
                $added = @($snapshot | Where-Object {
                    $candidate = $_
                    $_.ProcessId -notin @($targets.ProcessId) -and @($targets | Where-Object {
                        $_.ProcessId -eq $candidate.ParentProcessId -and $_.CreationDate -le $candidate.CreationDate
                    }).Count -gt 0
                })
                $targets += $added
            } while ($added.Count)
            foreach ($target in $targets) {
                $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($target.ProcessId)"
                if ($current -and $current.CreationDate -eq $target.CreationDate) {
                    Stop-Process -Id $target.ProcessId -Force -ErrorAction SilentlyContinue
                }
            }
        }
        Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
        Write-Output 'Realtime stopped; Gateway and Worker continue running'
        return
    }
    if ($owned) {
        if ($state.port -ne $Port -or $state.listenHost -ne $ListenHost -or [bool]$state.http -ne [bool]$Http -or
            [string]$state.allowedOrigins -ne $AllowedOrigins) {
            throw 'Realtime is running with different options. Stop it before changing its listener configuration.'
        }
        Write-Output "Realtime already running: $($state.url)"
        return
    }
    if (-not (Test-Path -LiteralPath $python)) { throw 'Install runtime/requirements-native.txt in .venv first.' }
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $Port is already in use; no existing service was stopped."
    }
    $arguments = @($server, '--host', $ListenHost, '--port', [string]$Port)
    if ($Http) { $arguments += '--http' }
    if ($AllowedOrigins) { $arguments += @('--allowed-origins', $AllowedOrigins) }
    if ($Action -eq 'run') {
        & $python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Realtime exited with status $LASTEXITCODE" }
        return
    }
    # Python paths may contain spaces; Start-Process joins ArgumentList itself.
    $quoted = @($arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' })
    $process = Start-Process -FilePath $python -ArgumentList $quoted -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logRoot 'realtime.stdout.log') -RedirectStandardError (Join-Path $logRoot 'realtime.stderr.log')
    $identity = Get-CimInstance Win32_Process -Filter "ProcessId = $($process.Id)"
    if (-not $identity) { throw "Realtime exited; see $logRoot/realtime.stderr.log" }
    $config = Get-Content -LiteralPath (Join-Path $projectRoot 'runtime/config.json') -Raw | ConvertFrom-Json
    $scheme = if (-not $Http -and $config.service.gateway_https) { 'wss' } else { 'ws' }
    $displayHost = if ($ListenHost -eq '0.0.0.0') { [string]$config.service.gateway_host } else { $ListenHost }
    $state = @{ processId = $process.Id; created = $identity.CreationDate.ToUniversalTime().Ticks.ToString()
                port = $Port; listenHost = $ListenHost; http = [bool]$Http; allowedOrigins = $AllowedOrigins
                url = "${scheme}://${displayHost}:$Port/v1/realtime?mode=audio&model=MiniCPM-o-4.5-Realtime" }
    $state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if (-not (Read-OwnedProcess $state)) { throw "Realtime exited; see $logRoot/realtime.stderr.log" }
        if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { $ready = $true; break }
        Start-Sleep -Milliseconds 250
    }
    if (-not $ready) { throw "Realtime is not listening; see $logRoot/realtime.stderr.log" }
    Write-Output "Realtime started: $($state.url)"
    Write-Output "Key file: $(Join-Path $stateRoot 'api-key')"
} finally {
    if ($controlLock) { $controlLock.Dispose() }
}
