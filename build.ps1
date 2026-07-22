# build.ps1 — compila los ejemplos de qubit.
# CPU (siempre disponible):   .\build.ps1
# GPU (requiere CUDA):        .\build.ps1 -Gpu
param([switch]$Gpu)

$root = $PSScriptRoot
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
New-Item -ItemType Directory -Force "$root\bin" | Out-Null

$examples = @("bell", "grover", "benchmark", "clusters")

if ($Gpu) {
    foreach ($ex in $examples) {
        Write-Host "nvcc: $ex (GPU)" -ForegroundColor Cyan
        cmd /c "call `"$vcvars`" >nul 2>&1 && nvcc -O2 -std=c++17 -arch=sm_86 -DQUBIT_CUDA -Xcompiler `"/openmp /utf-8 /EHsc`" -I `"$root\include`" `"$root\examples\$ex.cpp`" `"$root\src\qubit_gpu.cu`" -o `"$root\bin\$ex`_gpu.exe`""
        if ($LASTEXITCODE -ne 0) { Write-Host "FALLO: $ex" -ForegroundColor Red; exit 1 }
    }
    Write-Host "OK -> bin\*_gpu.exe" -ForegroundColor Green
} else {
    foreach ($ex in $examples) {
        Write-Host "cl: $ex (CPU)" -ForegroundColor Cyan
        cmd /c "call `"$vcvars`" >nul 2>&1 && cd /d `"$root\bin`" && cl /nologo /EHsc /std:c++17 /O2 /openmp /utf-8 /I `"$root\include`" `"$root\examples\$ex.cpp`" /Fe:$ex`_cpu.exe"
        if ($LASTEXITCODE -ne 0) { Write-Host "FALLO: $ex" -ForegroundColor Red; exit 1 }
    }
    Write-Host "OK -> bin\*_cpu.exe" -ForegroundColor Green
}
