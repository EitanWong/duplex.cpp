param(
    [string]$CudaRoot = $env:CUDA_PATH,
    [ValidateRange(5, 1800)][int]$TimeoutSeconds = 180,
    [switch]$OpenBrowser
)
& "$PSScriptRoot/service.ps1" -Action start -CudaRoot $CudaRoot -TimeoutSeconds $TimeoutSeconds -OpenBrowser:$OpenBrowser
exit $LASTEXITCODE
