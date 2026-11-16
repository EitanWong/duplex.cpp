param(
    [Parameter(Mandatory = $true)][ValidateSet('start', 'stop', 'status')][string]$Action,
    [string]$CudaRoot = $env:CUDA_PATH,
    [ValidateRange(5, 1800)][int]$TimeoutSeconds = 180,
    [switch]$OpenBrowser
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$runtimeRoot = Join-Path $projectRoot 'runtime'
$stateRoot = Join-Path $projectRoot '.local/state'
$statePath = Join-Path $stateRoot 'windows.json'
$lockPath = Join-Path $stateRoot 'windows.lock'
$launcherPath = Join-Path $projectRoot 'scripts/windows/run.ps1'
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$configPath = Join-Path $runtimeRoot 'config.json'
$logRoot = Join-Path $projectRoot 'logs'
$controlLock = $null
$http = $null
$newLaunchers = @()

function Normalize-Path([string]$Value) {
    if (-not $Value) { return '' }
    return [IO.Path]::GetFullPath($Value).Replace('\', '/').TrimEnd('/').ToLowerInvariant()
}

function Get-Identity($Process) {
    return [pscustomobject]@{
        id = [int]$Process.ProcessId
        created = $Process.CreationDate.ToUniversalTime().Ticks.ToString()
    }
}

function Test-Identity($Process, $Identity) {
    return $Process -and [int]$Process.ProcessId -eq [int]$Identity.id -and
        $Process.CreationDate.ToUniversalTime().Ticks.ToString() -eq [string]$Identity.created
}

function Get-OwnedProcesses($Snapshot) {
    $owned = @{}
    $byId = @{}
    foreach ($process in $Snapshot) { $byId[[int]$process.ProcessId] = $process }
    $venv = Normalize-Path $pythonPath
    foreach ($process in $Snapshot) {
        $command = ([string]$process.CommandLine).Replace('\', '/')
        foreach ($role in @('gateway', 'worker')) {
            $port = if ($role -eq 'gateway') { $gatewayPort } else { $workerPort }
            $scriptName = [regex]::Escape("$role.py")
            $absoluteScript = [regex]::Escape((Join-Path $runtimeRoot "$role.py").Replace('\', '/'))
            $scriptPattern = '(?i)(?:^|[\s"])(?:' + $scriptName + '|' + $absoluteScript + ')(?=[\s"]|$)'
            $portPattern = '(?:^|\s)--port(?:\s+|=)"?' + $port + '(?=[\s"]|$)'
            if ((Normalize-Path $process.ExecutablePath) -eq $venv -and
                $command -match $scriptPattern -and $command -match $portPattern) {
                $owned[[int]$process.ProcessId] = $role
            }
        }
        # Recover an orphaned native child after an interrupted controller.
        if ($nativePaths -contains (Normalize-Path $process.ExecutablePath) -and
            $command -match ('(?:^|\s)--port(?:\s+|=)"?' + $nativePort + '(?=[\s"]|$)')) {
            $owned[[int]$process.ProcessId] = 'worker'
        }
        foreach ($saved in @($state.launchers)) {
            if ((Test-Identity $process $saved) -and
                $process.Name -in @('pwsh.exe', 'powershell.exe') -and
                $command.IndexOf($launcherPath.Replace('\', '/'), [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $owned[[int]$process.ProcessId] = [string]$saved.role
            }
        }
    }
    # Windows venv's python.exe can be a shim; include its interpreter and engine.
    do {
        $added = $false
        foreach ($process in $Snapshot) {
            $processId = [int]$process.ProcessId
            if ($processId -ne $PID -and -not $owned.ContainsKey($processId) -and
                $owned.ContainsKey([int]$process.ParentProcessId) -and
                $byId[[int]$process.ParentProcessId].CreationDate -le $process.CreationDate) {
                # Parent IDs survive parent exit and may be reused by Windows.
                # An older process cannot be a child of this owned incarnation.
                $owned[$processId] = $owned[[int]$process.ParentProcessId]
                $added = $true
            }
        }
    } while ($added)
    return $owned
}

function Stop-OwnedProcesses($Snapshot, $Owned) {
    # Stop parents before descendants so a worker watchdog cannot restart CUDA.
    foreach ($role in @('gateway', 'worker')) {
        $remaining = @($Snapshot | Where-Object { $Owned[[int]$_.ProcessId] -eq $role })
        while ($remaining.Count) {
            $ids = @($remaining | ForEach-Object { [int]$_.ProcessId })
            $roots = @($remaining | Where-Object { [int]$_.ParentProcessId -notin $ids })
            if (-not $roots.Count) { throw 'Unexpected process-tree cycle.' }
            foreach ($process in $roots) {
                $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($process.ProcessId)"
                if (Test-Identity $current (Get-Identity $process)) {
                    $target = $null
                    try {
                        $target = Get-Process -Id $process.ProcessId -ErrorAction Stop
                        Stop-Process -InputObject $target -Force -ErrorAction Stop
                        if (-not $target.WaitForExit(10000)) { throw "Process $($process.ProcessId) did not exit." }
                    } catch {
                        # Stopping the parent can make a child exit between CIM
                        # inspection and Stop-Process. An exited child is success.
                        $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($process.ProcessId)"
                        if (Test-Identity $current (Get-Identity $process)) { throw }
                    } finally {
                        if ($target) { $target.Dispose() }
                    }
                }
            }
            $rootIds = @($roots | ForEach-Object { [int]$_.ProcessId })
            $remaining = @($remaining | Where-Object { [int]$_.ProcessId -notin $rootIds })
        }
    }
}

function Read-Health([string]$Url) {
    try {
        $response = $http.GetAsync($Url).GetAwaiter().GetResult()
        try {
            if (-not $response.IsSuccessStatusCode) { return $null }
            return ($response.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json)
        } finally { $response.Dispose() }
    } catch { return $null }
}

function Wait-Ready([string]$Role) {
    $timer = [Diagnostics.Stopwatch]::StartNew()
    while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        $snapshot = @(Get-CimInstance Win32_Process)
        $owned = Get-OwnedProcesses $snapshot
        if ($Role -notin @($owned.Values)) {
            throw "$Role exited. See $logRoot/windows-$Role*.log"
        }
        if ($Role -eq 'worker') {
            $health = Read-Health "$workerUrl/health"
            if ($health.status -eq 'healthy' -and $health.model_loaded) { return }
        } else {
            $health = Read-Health "$gatewayUrl/health"
            $workers = Read-Health "$gatewayUrl/workers"
            $ready = @($workers.workers | Where-Object {
                $_.port -eq $workerPort -and $_.status -in @('idle', 'busy')
            })
            if ($health.status -eq 'healthy' -and $ready.Count) { return }
        }
        Start-Sleep -Milliseconds 300
    }
    throw "$Role not ready after $TimeoutSeconds seconds. See $logRoot/windows-$Role*.log"
}

function Save-State {
    $state | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statePath -Encoding UTF8
}

function Start-Role([string]$Role, [string]$ResolvedCuda) {
    $shellPath = Join-Path $PSHOME 'pwsh.exe'
    if (-not (Test-Path -LiteralPath $shellPath)) { $shellPath = Join-Path $PSHOME 'powershell.exe' }
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $stdout = Join-Path $logRoot "windows-$Role-$stamp.log"
    $stderr = Join-Path $logRoot "windows-$Role-$stamp.err.log"
    # Start-Process on Windows PowerShell 5.1 can leak the controller's pipe
    # handles. Python passes only explicit stdio handles to a hidden process.
    $arguments = @('--shell', $shellPath, '--launcher', $launcherPath, '--role', $Role,
                    '--stdout', $stdout, '--stderr', $stderr)
    if ($ResolvedCuda) { $arguments += @('--cuda-root', $ResolvedCuda) }
    $launchedId = & $pythonPath (Join-Path $projectRoot 'scripts/windows/detach.py') @arguments
    if ($LASTEXITCODE -ne 0) { throw "$Role process launcher failed." }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$launchedId)"
    if (-not $process) { throw "$Role failed to start. See $stderr" }
    $identity = Get-Identity $process
    $identity | Add-Member NoteProperty role $Role
    $identity | Add-Member NoteProperty stdout $stdout
    $identity | Add-Member NoteProperty stderr $stderr
    $script:newLaunchers += $identity
    $state.launchers = @($state.launchers) + @($identity)
    Save-State
    Write-Host "Starting $Role; log: $stderr"
}

function Resolve-Cuda {
    if ($CudaRoot) { return (Resolve-Path -LiteralPath $CudaRoot).Path }
    $cache = Join-Path $projectRoot 'engine/build-cuda/CMakeCache.txt'
    if (Test-Path -LiteralPath $cache) {
        $match = Select-String -LiteralPath $cache -Pattern '^CUDAToolkit_ROOT:[^=]+=(.+)$' | Select-Object -First 1
        if ($match -and (Test-Path -LiteralPath $match.Matches[0].Groups[1].Value)) {
            return $match.Matches[0].Groups[1].Value
        }
    }
    # DLLs may already be on PATH when no toolkit root is supplied.
    return ''
}

try {
    if ($env:OS -ne 'Windows_NT') { throw 'Use scripts/macos/ on macOS. Linux serving is not validated.' }
    if ($Action -ne 'status') {
        New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
        foreach ($extension in @('json', 'lock')) {
            $legacy = Join-Path $projectRoot "service-windows.$extension"
            $destination = Join-Path $stateRoot "windows.$extension"
            if ((Test-Path -LiteralPath $legacy) -and -not (Test-Path -LiteralPath $destination)) {
                Move-Item -LiteralPath $legacy -Destination $destination
            }
        }
        try { $controlLock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') }
        catch { throw 'Another Windows start/stop operation is running. Retry when it finishes.' }
    }
    $config = $null
    if (Test-Path -LiteralPath $configPath) {
        $config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    $state = [pscustomobject]@{ launchers = @() }
    if (Test-Path -LiteralPath $statePath) {
        try { $state = Get-Content -LiteralPath $statePath -Raw -Encoding UTF8 | ConvertFrom-Json }
        catch { Write-Warning 'Ignoring invalid Windows state; checking project processes directly.' }
    }
    $workerPort = if ($config.service.worker_base_port) { [int]$config.service.worker_base_port } elseif ($state.worker_port) { [int]$state.worker_port } else { 22440 }
    $gatewayPort = if ($config.service.gateway_port) { [int]$config.service.gateway_port } elseif ($state.gateway_port) { [int]$state.gateway_port } else { 8040 }
    $nativePort = if ($config.cpp_backend.cpp_server_port) { [int]$config.cpp_backend.cpp_server_port } elseif ($state.native_port) { [int]$state.native_port } else { 19060 }
    $gatewayHost = if ($config.service.gateway_host) { [string]$config.service.gateway_host } else { '127.0.0.1' }
    if ($gatewayHost -in @('0.0.0.0', '::')) { $gatewayHost = 'localhost' }
    $scheme = if ($config.service.gateway_https) { 'https' } else { 'http' }
    $gatewayUrl = '{0}://{1}:{2}' -f $scheme, $gatewayHost, $gatewayPort
    $workerUrl = "http://127.0.0.1:$workerPort"
    $engineRoot = if ($config.cpp_backend.llamacpp_root) { [string]$config.cpp_backend.llamacpp_root } else { Join-Path $projectRoot 'engine' }
    $binaryDirs = @((Join-Path $engineRoot 'build-cuda/bin'), (Join-Path $engineRoot 'build/bin/Release'), (Join-Path $engineRoot 'build/bin'))
    if ($config.cpp_backend.native_binary_dir) { $binaryDirs += [string]$config.cpp_backend.native_binary_dir }
    if ($env:OMNI_NATIVE_RELEASE_DIR) { $binaryDirs += $env:OMNI_NATIVE_RELEASE_DIR }
    $nativePaths = @($binaryDirs | ForEach-Object {
        Normalize-Path (Join-Path $_ 'llama-omni-server.exe')
        Normalize-Path (Join-Path $_ 'llama-server.exe')
    })
    $snapshot = @(Get-CimInstance Win32_Process)
    $owned = Get-OwnedProcesses $snapshot

    if ($Action -eq 'stop') {
        Stop-OwnedProcesses $snapshot $owned
        # Confirm all original process identities are gone, including CUDA children.
        $timer = [Diagnostics.Stopwatch]::StartNew()
        do {
            $remaining = @($snapshot | Where-Object {
                if (-not $owned.ContainsKey([int]$_.ProcessId)) { return $false }
                $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($_.ProcessId)"
                Test-Identity $current (Get-Identity $_)
            })
            if (-not $remaining.Count) { break }
            Start-Sleep -Milliseconds 100
        } while ($timer.Elapsed.TotalSeconds -lt 10)
        if ($remaining.Count) { throw "Project processes still exiting: $($remaining.ProcessId -join ', ')" }
        # A watchdog may have spawned one last native child during the first
        # snapshot. Its exact executable and configured port still identify it.
        $snapshot = @(Get-CimInstance Win32_Process)
        $leftovers = Get-OwnedProcesses $snapshot
        if ($leftovers.Count) { Stop-OwnedProcesses $snapshot $leftovers }
        Remove-Item -LiteralPath $statePath -ErrorAction SilentlyContinue
        Write-Host "Stopped $($owned.Count) project processes. GPU memory is released when the native engine exits."
        exit 0
    }

    Add-Type -AssemblyName System.Net.Http
    $handler = New-Object System.Net.Http.HttpClientHandler
    $handler.UseProxy = $false
    $http = New-Object System.Net.Http.HttpClient($handler)
    $http.Timeout = [TimeSpan]::FromSeconds(2)

    if ($Action -eq 'status') {
        foreach ($role in @('worker', 'gateway')) {
            $ids = @($owned.Keys | Where-Object { $owned[$_] -eq $role } | Sort-Object)
            $url = if ($role -eq 'worker') { $workerUrl } else { $gatewayUrl }
            $health = Read-Health "$url/health"
            $description = if ($health) { $health | ConvertTo-Json -Compress } else { 'unavailable (stopped, loading, or TLS trust required)' }
            Write-Host "$role PIDs: $($ids -join ', '); $description"
        }
        Write-Host "Page: $gatewayUrl/omni"
        Write-Host "Logs: $logRoot/windows-*.log"
        exit 0
    }

    if (-not $config -or $config.backend -ne 'cpp') { throw 'Configure runtime/config.json for the C++ backend first.' }
    if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create .venv and install runtime/requirements-native.txt first.' }
    if (-not @($nativePaths | Where-Object { Test-Path -LiteralPath $_ }).Count) { throw 'Build the native Windows engine with scripts/windows/build.ps1 first.' }
    # Refuse unrelated port owners, rather than starting duplicates or killing them.
    $listeners = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object {
        $_.LocalPort -in @($workerPort, $gatewayPort, $nativePort)
    })
    foreach ($listener in $listeners) {
        if (-not $owned.ContainsKey([int]$listener.OwningProcess)) {
            throw "Port $($listener.LocalPort) belongs to unrelated PID $($listener.OwningProcess). Resolve the conflict first."
        }
    }
    New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
    $state = [pscustomobject]@{
        worker_port = $workerPort; gateway_port = $gatewayPort; native_port = $nativePort
        launchers = @($state.launchers | Where-Object {
            $saved = $_
            @($snapshot | Where-Object { Test-Identity $_ $saved }).Count -gt 0
        })
    }
    Save-State
    $resolvedCuda = Resolve-Cuda
    foreach ($role in @('worker', 'gateway')) {
        $roleProcesses = @($snapshot | Where-Object { $owned[[int]$_.ProcessId] -eq $role })
        if ($role -eq 'worker' -and $roleProcesses.Count -and
            -not @($roleProcesses | Where-Object { $_.Name -in @('python.exe', 'pwsh.exe', 'powershell.exe') }).Count) {
            throw 'Found an orphaned native engine. Run scripts/windows/stop.ps1, then start again.'
        }
        if ($roleProcesses.Count) { Write-Host "Reusing running $role." }
        else { Start-Role $role $resolvedCuda }
        Wait-Ready $role
        Write-Host "$role ready."
    }
    Write-Host "Ready: $gatewayUrl/omni"
    Write-Host 'Stop: scripts/windows/stop.bat (or stop.ps1)'
    if ($OpenBrowser) { Start-Process "$gatewayUrl/omni" }
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    # Roll back only processes launched by this invocation; preserve reused services.
    if ($newLaunchers.Count) {
        $snapshot = @(Get-CimInstance Win32_Process)
        $state.launchers = $newLaunchers
        $owned = Get-OwnedProcesses $snapshot
        $newOwned = @{}
        $newRoles = @($newLaunchers | ForEach-Object { $_.role })
        foreach ($processId in $owned.Keys) {
            # A failed launcher may already have exited, leaving only the
            # identifiable Python shim or native engine. Roles were launched
            # only when absent, so reused roles are excluded from rollback.
            if ($owned[$processId] -in $newRoles) { $newOwned[$processId] = $owned[$processId] }
        }
        Stop-OwnedProcesses $snapshot $newOwned
    }
    exit 1
} finally {
    if ($http) { $http.Dispose() }
    if ($controlLock) { $controlLock.Dispose() }
}
