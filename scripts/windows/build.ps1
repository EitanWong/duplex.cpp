param(
    [string]$CudaRoot = $env:CUDA_PATH,
    [string]$VsDevCmd,
    [string]$WindowsSdkRoot,
    [int]$Jobs = 8,
    [switch]$AllKvQuants,
    [switch]$ConfigureOnly
)
$ErrorActionPreference = 'Stop'
$env:VSLANG = '1033'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $CudaRoot -or -not (Test-Path (Join-Path $CudaRoot 'bin/nvcc.exe'))) {
    throw 'Set -CudaRoot to a CUDA 12.8+ toolkit with nvcc, cuBLAS and runtime libraries.'
}
if ($Jobs -lt 1) { throw 'Jobs must be positive.' }
if ($VsDevCmd) {
    if (-not (Test-Path -LiteralPath $VsDevCmd)) { throw 'VsDevCmd not found.' }
    # Import the developer environment only into this process, leaving the system unchanged.
    $developerEnv = & $env:ComSpec /d /s /c "`"`"$VsDevCmd`" -arch=x64 -host_arch=x64 >nul && set`""
    if ($LASTEXITCODE -ne 0) { throw 'Visual Studio environment initialization failed.' }
    foreach ($line in $developerEnv) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw 'Use a VS 2022 developer terminal or supply -VsDevCmd.'
}
if ($WindowsSdkRoot) {
    $sdkVersion = Get-ChildItem (Join-Path $WindowsSdkRoot 'Lib') -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName 'um/x64/kernel32.lib') } |
        Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1 -ExpandProperty Name
    if (-not $sdkVersion) { throw 'No x64 Windows SDK libraries found.' }
    $env:WindowsSdkDir = $WindowsSdkRoot
    $env:WindowsSDKVersion = "$sdkVersion\"
    $env:UniversalCRTSdkDir = $WindowsSdkRoot
    $env:UCRTVersion = $sdkVersion
    $env:PATH = "$WindowsSdkRoot/bin/$sdkVersion/x64;$env:PATH"
    $env:INCLUDE = "$WindowsSdkRoot/Include/$sdkVersion/ucrt;$WindowsSdkRoot/Include/$sdkVersion/shared;$WindowsSdkRoot/Include/$sdkVersion/um;$WindowsSdkRoot/Include/$sdkVersion/winrt;$env:INCLUDE"
    $env:LIB = "$WindowsSdkRoot/Lib/$sdkVersion/ucrt/x64;$WindowsSdkRoot/Lib/$sdkVersion/um/x64;$env:LIB"
}
$env:CUDA_PATH = (Resolve-Path -LiteralPath $CudaRoot).Path
$env:PATH = "$projectRoot/.venv/Scripts;$env:CUDA_PATH/bin;$env:PATH"
$cmake = (Get-Command cmake.exe -ErrorAction Stop).Source
$buildDir = Join-Path $projectRoot 'engine/build-cuda'
$dependencyRepairStamp = Join-Path $buildDir '.msvc-header-deps-repaired'
$allKvQuantsValue = if ($AllKvQuants) { 'ON' } else { 'OFF' }
& $cmake -S "$projectRoot/engine" -B $buildDir -G Ninja `
    '-DCMAKE_BUILD_TYPE=Release' '-DCMAKE_CUDA_ARCHITECTURES=86-real' `
    "-DCUDAToolkit_ROOT=$env:CUDA_PATH" "-DCMAKE_CUDA_COMPILER=$env:CUDA_PATH/bin/nvcc.exe" `
    '-DGGML_CUDA=ON' '-DGGML_CUDA_GRAPHS=ON' '-DGGML_CUDA_FA=ON' `
    "-DGGML_CUDA_FA_ALL_QUANTS=$allKvQuantsValue" `
    '-DGGML_NATIVE=OFF' '-DGGML_AVX2=ON' '-DGGML_AVX512=OFF' `
    '-DGGML_METAL=OFF' '-DENABLE_COREML=OFF' '-DGGML_BLAS=OFF' `
    '-DLLAMA_OPENSSL=OFF' '-DLLAMA_BUILD_TESTS=ON' '-DLLAMA_BUILD_EXAMPLES=OFF' `
    '-DLLAMA_BUILD_UI=OFF' '-DLLAMA_BUILD_APP=OFF'
if ($LASTEXITCODE -ne 0) { throw 'CUDA configuration failed.' }
# CMake can decode UTF-8 MSVC /showIncludes output as OEM text on Windows.
# Ninja then silently records zero header dependencies, leaving incompatible
# objects after a header edit. Repair that generated detection and regenerate.
$prefixRepaired = $false
$oemEncoding = [Text.Encoding]::GetEncoding(437, [Text.EncoderFallback]::ExceptionFallback, [Text.DecoderFallback]::ExceptionFallback)
$strictUtf8 = [Text.UTF8Encoding]::new($false, $true)
Get-ChildItem -Path "$buildDir/CMakeFiles/*/CMake*CCompiler.cmake", "$buildDir/CMakeFiles/*/CMakeCXXCompiler.cmake" | ForEach-Object {
    $compilerFile = $_.FullName
    $compilerText = [IO.File]::ReadAllText($compilerFile)
    $prefixMatch = [regex]::Match($compilerText, 'set\(CMAKE_C(?:XX)?_CL_SHOWINCLUDES_PREFIX "([^"]+)"\)')
    if ($prefixMatch.Success) {
        try { $decodedPrefix = $strictUtf8.GetString($oemEncoding.GetBytes($prefixMatch.Groups[1].Value)) }
        catch { $decodedPrefix = $prefixMatch.Groups[1].Value }
        if ($decodedPrefix -ne $prefixMatch.Groups[1].Value) {
            $compilerText = $compilerText.Replace($prefixMatch.Groups[1].Value, $decodedPrefix)
            [IO.File]::WriteAllText($compilerFile, $compilerText, [Text.UTF8Encoding]::new($false))
            $prefixRepaired = $true
        }
    }
}
if ($prefixRepaired) {
    if (Test-Path -LiteralPath $dependencyRepairStamp) {
        Remove-Item -LiteralPath $dependencyRepairStamp -Force
    }
    & $cmake -S "$projectRoot/engine" -B $buildDir
    if ($LASTEXITCODE -ne 0) { throw 'MSVC header dependency repair failed.' }
}
if (-not $ConfigureOnly) {
    # Include every previously built native consumer, including standalone
    # tools and Token2Wav wrappers. A repaired prefix cannot retroactively fill
    # old dependency records; retaining such objects can corrupt class layouts.
    if ($prefixRepaired -or -not (Test-Path -LiteralPath $dependencyRepairStamp)) {
        $ninja = (Get-Command ninja.exe -ErrorAction Stop).Source
        $dependencyLines = & $ninja -C $buildDir -t deps
        if ($LASTEXITCODE -ne 0) { throw 'Could not inspect Ninja header dependencies.' }
        foreach ($line in $dependencyLines) {
            if ($line -match '^(.+\.(?:c|cc|cpp|cxx)\.obj): #deps 0,') {
                $objectPath = [IO.Path]::GetFullPath((Join-Path $buildDir $Matches[1]))
                $buildPrefix = [IO.Path]::GetFullPath($buildDir).TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
                if (-not $objectPath.StartsWith($buildPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                    throw 'Ninja object path is outside the build directory.'
                }
                if (Test-Path -LiteralPath $objectPath) { Remove-Item -LiteralPath $objectPath -Force }
            }
        }
    }
    & $cmake --build $buildDir --target llama-omni-server llama-bench token2wav-example test-backend-ops test-kv-budget test-kv-ledger test-tts-cpu-head test-tts-cuda-head test-vision-encode test-audio-encode test-tts-condition --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw 'CUDA build failed.' }
    [IO.File]::WriteAllText($dependencyRepairStamp, 'MSVC include dependency repair completed.')
}
Write-Output "CUDA build directory: $buildDir"
