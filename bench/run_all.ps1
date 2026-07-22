# Full benchmark run: qubit (all backends) vs Qiskit Aer.
# Writes CSVs to bench\results\. Run after building with build.ps1 (and
# nvcc for the GPU rows).
param([int]$MaxQubits = 26)

$root = "D:\Github\qubit"
$out = "$root\bench\results"
New-Item -ItemType Directory -Force $out | Out-Null

Write-Host "== qubit backends =="
& "$root\bin\benchsuite.exe" $MaxQubits auto 5   | Out-File -Encoding utf8 "$out\qubit_auto.csv"
& "$root\bin\benchsuite.exe" $MaxQubits dense 5  | Out-File -Encoding utf8 "$out\qubit_dense.csv"
& "$root\bin\benchsuite.exe" $MaxQubits groups 5 | Out-File -Encoding utf8 "$out\qubit_groups.csv"
& "$root\bin\benchsuite.exe" $MaxQubits blocks 5 | Out-File -Encoding utf8 "$out\qubit_blocks.csv"
if (Test-Path "$root\bin\benchsuite_gpu.exe") {
    Write-Host "== qubit GPU =="
    & "$root\bin\benchsuite_gpu.exe" $MaxQubits gpu 5 | Out-File -Encoding utf8 "$out\qubit_gpu.csv"
}

Write-Host "== Qiskit Aer =="
py "$root\bench\bench_qiskit.py" $MaxQubits statevector | Out-File -Encoding utf8 "$out\aer_sv.csv"
py "$root\bench\bench_qiskit.py" $MaxQubits matrix_product_state | Out-File -Encoding utf8 "$out\aer_mps.csv"

Write-Host "done -> $out"
