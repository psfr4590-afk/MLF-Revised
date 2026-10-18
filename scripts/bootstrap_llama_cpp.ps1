param([switch]$Cuda)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Target = Join-Path $Root "third_party\llama.cpp"
$Repo = "https://github.com/ggml-org/llama.cpp.git"
$Tag = "b10516"
$Commit = "b95502b"

Write-Host "`n=== LLAMA.CPP BOOTSTRAP ===" -ForegroundColor Cyan

if (-not (Test-Path (Join-Path $Target ".git"))) {
    if (Test-Path $Target) {
        throw "Refusing to delete or overwrite an existing directory without a Git checkout: $Target. Inspect it and move or back it up manually before retrying."
    }

    New-Item -ItemType Directory -Force -Path (Split-Path $Target) | Out-Null

    git clone --depth 1 --branch $Tag $Repo $Target
    if ($LASTEXITCODE -ne 0) {
        throw "llama.cpp clone failed with exit code $LASTEXITCODE"
    }
}

Set-Location $Target

$Actual = git rev-parse HEAD
if ($LASTEXITCODE -ne 0) {
    throw "Unable to read the llama.cpp checkout commit"
}
$Actual = $Actual.Trim()

if (-not $Actual.StartsWith($Commit)) {
    throw "llama.cpp pin mismatch: expected $Commit got $Actual"
}

if (-not (Test-Path "convert_hf_to_gguf.py")) {
    throw "Missing convert_hf_to_gguf.py"
}

$Build = "build-model-lab"

if ($Cuda) {
    Write-Host "CUDA native build requested. CMake will fail closed if the CUDA toolkit/compiler is unavailable." -ForegroundColor Yellow
} else {
    Write-Host "Using the portable CPU native build profile. Pass -Cuda to enable GGML_CUDA." -ForegroundColor Cyan
}

Write-Host "`n=== CONFIGURING CMAKE ===" -ForegroundColor Cyan

$CudaValue = if ($Cuda) { "ON" } else { "OFF" }
$Args = @(
    "-S", ".",
    "-B", $Build,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DGGML_NATIVE=OFF",
    "-DGGML_OPENMP=OFF",
    "-DGGML_LLAMAFILE=OFF",
    "-DLLAMA_CURL=OFF",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_SERVER=OFF",
    "-DLLAMA_BUILD_APP=ON",
    "-DLLAMA_BUILD_TOOLS=ON",
    "-DLLAMA_BUILD_EXAMPLES=ON",
    "-DGGML_CUDA=$CudaValue"
)

cmake @Args
if ($LASTEXITCODE -ne 0) {
    throw "CMake configuration failed with exit code $LASTEXITCODE"
}

Write-Host "`n=== BUILDING LLAMA-QUANTIZE ===" -ForegroundColor Cyan

# Clean the generated build outputs first so a failed build cannot be
# mistaken for success merely because an older executable is still present.
cmake --build $Build --config Release --target llama-quantize --clean-first --parallel 2
if ($LASTEXITCODE -ne 0) {
    throw "llama-quantize build failed with exit code $LASTEXITCODE"
}

Write-Host "`n=== BUILDING LLAMA-CLI ===" -ForegroundColor Cyan

$CachePath = Join-Path $Build "CMakeCache.txt"
$GeneratorLine = Get-Content $CachePath |
    Where-Object { $_ -match "^CMAKE_GENERATOR:INTERNAL=" } |
    Select-Object -First 1

if ($GeneratorLine -match "Visual Studio") {
    $CliProject = Get-ChildItem -Path $Build -Recurse -File -Filter "llama-cli.vcxproj" |
        Select-Object -First 1

    if (-not $CliProject) {
        throw "Visual Studio build is missing the generated llama-cli.vcxproj"
    }

    $VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $VsWhere)) {
        throw "Cannot locate Visual Studio vswhere.exe to resolve MSBuild"
    }

    $MsBuild = & $VsWhere -latest -products "*" `
        -requires Microsoft.Component.MSBuild `
        -find "MSBuild\**\Bin\amd64\MSBuild.exe" |
        Select-Object -First 1

    if (-not $MsBuild -or -not (Test-Path $MsBuild)) {
        throw "Cannot locate the Visual Studio MSBuild executable"
    }

    & $MsBuild $CliProject.FullName `
        /p:Configuration=Release /p:Platform=x64 /m:2 /v:minimal

    if ($LASTEXITCODE -ne 0) {
        throw "llama-cli MSBuild failed with exit code $LASTEXITCODE"
    }
}
else {
    cmake --build $Build --config Release --target llama-cli --parallel 2
    if ($LASTEXITCODE -ne 0) {
        throw "llama-cli build failed with exit code $LASTEXITCODE"
    }
}
$Quant = Get-ChildItem -Path $Build `
    -Recurse `
    -File `
    -Filter "llama-quantize.exe" |
    Select-Object -First 1

$Cli = Get-ChildItem -Path $Build `
    -Recurse `
    -File `
    -Filter "llama-cli.exe" |
    Select-Object -First 1

if (-not $Quant) {
    throw "llama-quantize.exe was not produced by the successful build"
}

if (-not $Cli) {
    throw "llama-cli.exe was not produced by the successful build"
}

Write-Host "`n=== LLAMA.CPP READY ===" -ForegroundColor Green
Write-Host "  converter: $Target\convert_hf_to_gguf.py"
Write-Host "  quantizer: $($Quant.FullName)"
Write-Host "  cli:       $($Cli.FullName)"

Write-Host "`n=== VERIFYING LLAMA-CLI AND QUANTIZER ===" -ForegroundColor Cyan
& $Cli.FullName --help *> $null
if ($LASTEXITCODE -ne 0) {
    throw "llama-cli --help failed with exit code $LASTEXITCODE"
}
& $Quant.FullName --help *> $null
if ($LASTEXITCODE -ne 0) {
    throw "llama-quantize --help failed with exit code $LASTEXITCODE"
}

Write-Host "`n=== BUILDING LLAMA-APP ===" -ForegroundColor Cyan
cmake --build $Build --config Release --target llama-app --parallel 2
if ($LASTEXITCODE -ne 0) {
    throw "llama-app build failed with exit code $LASTEXITCODE"
}

$App = Get-ChildItem -Path $Build -Recurse -File -Filter "llama.exe" |
    Select-Object -First 1
if (-not $App) {
    $App = Get-ChildItem -Path $Build -Recurse -File -Filter "llama" |
        Select-Object -First 1
}
if (-not $App) {
    throw "llama-app build did not produce llama.exe or llama"
}

Write-Host "`n=== VERIFYING LLAMA-APP RUNTIME ===" -ForegroundColor Cyan
& $App.FullName --help *> $null
if ($LASTEXITCODE -ne 0) {
    throw "llama-app --help failed with exit code $LASTEXITCODE"
}
& $App.FullName --version
if ($LASTEXITCODE -ne 0) {
    throw "llama-app --version failed with exit code $LASTEXITCODE"
}

Write-Host "  app:       $($App.FullName)"
