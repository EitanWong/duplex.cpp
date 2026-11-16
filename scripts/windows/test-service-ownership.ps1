$ErrorActionPreference = 'Stop'
# Extract only the pure inventory helpers: never execute the service controller.
$syntaxTokens = $null
$syntaxErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $PSScriptRoot 'service.ps1'), [ref]$syntaxTokens, [ref]$syntaxErrors)
foreach ($name in @('Normalize-Path', 'Get-Identity', 'Test-Identity', 'Get-OwnedProcesses')) {
    $node = $ast.Find({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name }, $true)
    Invoke-Expression $node.Extent.Text
}
$pythonPath = 'C:/fixture/.venv/Scripts/python.exe'
$runtimeRoot = 'C:/fixture/runtime'
$launcherPath = 'C:/fixture/scripts/windows/run.ps1'
$workerPort = 22440
$gatewayPort = 8040
$nativePort = 19080
$nativePaths = @((Normalize-Path 'C:/fixture/llama-omni-server.exe'))
$state = [pscustomobject]@{ launchers = @() }
$started = [DateTime]::UtcNow
$parent = [pscustomobject]@{ ProcessId = 123; ParentProcessId = 99; CreationDate = $started; ExecutablePath = $pythonPath; CommandLine = 'python.exe worker.py --port 22440'; Name = 'python.exe' }
$child = [pscustomobject]@{ ProcessId = 124; ParentProcessId = 123; CreationDate = $started.AddSeconds(1); ExecutablePath = 'C:/python.exe'; CommandLine = 'python.exe worker.py --port 22440'; Name = 'python.exe' }
$old = [pscustomobject]@{ ProcessId = 125; ParentProcessId = 123; CreationDate = $started.AddSeconds(-20); ExecutablePath = 'C:/unrelated.exe'; CommandLine = 'unrelated.exe'; Name = 'unrelated.exe' }
$grandchild = [pscustomobject]@{ ProcessId = 126; ParentProcessId = 124; CreationDate = $started.AddSeconds(2); ExecutablePath = 'C:/fixture/llama-omni-server.exe'; CommandLine = 'llama-omni-server.exe --port 19080'; Name = 'llama-omni-server.exe' }
$owned = Get-OwnedProcesses @($parent, $child, $old, $grandchild)
if ($owned.Count -ne 3 -or $owned.ContainsKey(125) -or -not $owned.ContainsKey(126)) {
    throw 'Process ownership crossed a reused parent PID, or lost a real child.'
}
'PASS: process identities exclude children older than a reused parent PID'
