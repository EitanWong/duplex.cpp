param(
    [Parameter(Mandatory = $true)][int]$TargetProcessId,
    [Parameter(Mandatory = $true)][string]$OutputPath
)
$ErrorActionPreference = 'Stop'
$initial = Get-Process -Id $TargetProcessId
$started = $initial.StartTime
while ($true) {
    $process = Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue
    if (-not $process -or $process.StartTime -ne $started) { break }
    $gpu = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory |
        Where-Object { $_.Name -match "^pid_${TargetProcessId}_" })
    [pscustomobject]@{
        timestamp = [DateTime]::UtcNow.ToString('o')
        pid = $TargetProcessId
        working_set_bytes = $process.WorkingSet64
        private_bytes = $process.PrivateMemorySize64
        dedicated_bytes = if ($gpu.Count) { ($gpu | Measure-Object DedicatedUsage -Sum).Sum } else { $null }
        shared_bytes = if ($gpu.Count) { ($gpu | Measure-Object SharedUsage -Sum).Sum } else { $null }
        committed_bytes = if ($gpu.Count) { ($gpu | Measure-Object TotalCommitted -Sum).Sum } else { $null }
    } | Export-Csv -LiteralPath $OutputPath -NoTypeInformation -Append -Encoding UTF8
    Start-Sleep -Milliseconds 1000
}
