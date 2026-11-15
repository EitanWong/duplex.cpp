param(
    [ValidateSet('worker', 'gateway')][string]$Role = 'worker',
    [string]$CudaRoot = $env:CUDA_PATH
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Create .venv and install runtime/requirements-native.txt first.' }
$configPath = Join-Path $projectRoot 'runtime/config.json'
if (-not (Test-Path -LiteralPath $configPath)) { throw 'Run scripts/configure.py --profile nvidia-rtx3060 first.' }
$config = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($config.backend -ne 'cpp') { throw 'This launcher requires the C++ backend.' }
if ($CudaRoot) {
    $env:CUDA_PATH = (Resolve-Path -LiteralPath $CudaRoot).Path
    $env:PATH = "$env:CUDA_PATH/bin;$env:PATH"
}
if ($config.cpp_backend.runtime_env) {
    foreach ($property in $config.cpp_backend.runtime_env.PSObject.Properties) {
        if ($property.Name -notmatch '^(OMNI_|GGML_CUDA_|CUDA_|OMP_|OPENBLAS_|MKL_)') {
            throw "Unsupported inference setting: $($property.Name)"
        }
        if ($null -eq [Environment]::GetEnvironmentVariable($property.Name, 'Process')) {
            [Environment]::SetEnvironmentVariable($property.Name, [string]$property.Value, 'Process')
        }
    }
}
Push-Location (Join-Path $projectRoot 'runtime')
try {
    $workerPort = [int]$config.service.worker_base_port
    if ($Role -eq 'worker') {
        & $python worker.py --host 127.0.0.1 --port $workerPort --gpu-id 0 --worker-index 0
    } else {
        $gatewayHost = if ($config.service.gateway_host) { [string]$config.service.gateway_host } else { '127.0.0.1' }
        $gatewayArgs = @('gateway.py', '--host', $gatewayHost, '--port', [string]$config.service.gateway_port,
                         '--workers', "127.0.0.1:$workerPort")
        if ($config.service.gateway_https) {
            $certificatePath = if ($config.service.gateway_ssl_certfile) { [string]$config.service.gateway_ssl_certfile } else { 'certs/lan-cert.pem' }
            $privateKeyPath = if ($config.service.gateway_ssl_keyfile) { [string]$config.service.gateway_ssl_keyfile } else { 'certs/lan-key.pem' }
            if (-not (Test-Path -LiteralPath $certificatePath) -or -not (Test-Path -LiteralPath $privateKeyPath)) {
                throw 'TLS files missing. Run scripts/configure-lan.py --ip <LAN address> first.'
            }
            $gatewayArgs += @('--https', '--ssl-certfile', $certificatePath, '--ssl-keyfile', $privateKeyPath)
        } else {
            $gatewayArgs += '--http'
        }
        & $python @gatewayArgs
    }
    if ($LASTEXITCODE -ne 0) { throw "$Role exited with status $LASTEXITCODE" }
} finally {
    Pop-Location
}
