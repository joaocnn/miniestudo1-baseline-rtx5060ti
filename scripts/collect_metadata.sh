#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Uso: $0 <TIMESTAMP>" >&2
    exit 1
fi

TIMESTAMP="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/data/raw"
OUTPUT_FILE="$OUTPUT_DIR/metadata_${TIMESTAMP}.json"

mkdir -p "$OUTPUT_DIR"

echo "Coletando metadados para timestamp: $TIMESTAMP"

# GPU
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | xargs)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | xargs)
VBIOS_VERSION=$(nvidia-smi --query-gpu=vbios_version --format=csv,noheader | xargs)
GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | xargs)
CLOCK_BASE=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader | xargs)
CLOCK_BOOST=$(nvidia-smi --query-gpu=clocks.max.gr --format=csv,noheader | xargs)

# CUDA: versão do runtime via CuPy
CUDA_RUNTIME_VERSION=$(python3 -c "
import cupy
v = cupy.cuda.runtime.runtimeGetVersion()
major = v // 1000
minor = (v % 1000) // 10
print(f'{major}.{minor}')
")

# cuBLAS versão
CUBLAS_VERSION=$(python3 -c "
import cupy
h = cupy.cuda.cublas.create()
v = cupy.cuda.cublas.getVersion(h)
cupy.cuda.cublas.destroy(h)
print(v)
")

# Sistema
UNAME=$(uname -a)
DISTRO=$(lsb_release -a 2>/dev/null || (cat /etc/os-release 2>/dev/null) || echo "N/A")
KERNEL=$(uname -r)

# CPU: primeira ocorrência de "model name" em /proc/cpuinfo
CPU_MODEL=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs)

# RAM total
RAM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{print $2, $3}')

# Python e bibliotecas
PYTHON_VERSION=$(python3 --version 2>&1)
CUPY_VERSION=$(python3 -c "import cupy; print(cupy.__version__)")
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
PANDAS_VERSION=$(python3 -c "import pandas; print(pandas.__version__)" 2>/dev/null || echo "N/A")
SCIPY_VERSION=$(python3 -c "import scipy; print(scipy.__version__)" 2>/dev/null || echo "N/A")

# Saída completa de nvidia-smi
NVIDIA_SMI_FULL=$(nvidia-smi)

# Constrói o JSON com jq para garantir escaping correto
jq -n \
    --arg timestamp        "$TIMESTAMP" \
    --arg gpu_name         "$GPU_NAME" \
    --arg driver_version   "$DRIVER_VERSION" \
    --arg vbios_version    "$VBIOS_VERSION" \
    --arg memory_total     "$GPU_MEM_TOTAL" \
    --arg clock_base       "$CLOCK_BASE" \
    --arg clock_boost      "$CLOCK_BOOST" \
    --arg cuda_runtime     "$CUDA_RUNTIME_VERSION" \
    --arg cublas_version   "$CUBLAS_VERSION" \
    --arg uname            "$UNAME" \
    --arg distro           "$DISTRO" \
    --arg kernel           "$KERNEL" \
    --arg cpu_model        "$CPU_MODEL" \
    --arg ram_total        "$RAM_TOTAL" \
    --arg python_version   "$PYTHON_VERSION" \
    --arg cupy_version     "$CUPY_VERSION" \
    --arg numpy_version    "$NUMPY_VERSION" \
    --arg pandas_version   "$PANDAS_VERSION" \
    --arg scipy_version    "$SCIPY_VERSION" \
    --arg nvidia_smi_full  "$NVIDIA_SMI_FULL" \
    '{
        timestamp: $timestamp,
        gpu: {
            name:          $gpu_name,
            driver_version: $driver_version,
            vbios_version:  $vbios_version,
            memory_total:   $memory_total,
            clock_base:     $clock_base,
            clock_boost:    $clock_boost
        },
        cuda: {
            runtime_version: $cuda_runtime,
            cublas_version:  $cublas_version
        },
        system: {
            uname:  $uname,
            distro: $distro,
            kernel: $kernel
        },
        cpu: {
            model: $cpu_model
        },
        memory: {
            ram_total: $ram_total
        },
        python: {
            python:  $python_version,
            cupy:    $cupy_version,
            numpy:   $numpy_version,
            pandas:  $pandas_version,
            scipy:   $scipy_version
        },
        nvidia_smi_full: $nvidia_smi_full
    }' > "$OUTPUT_FILE"

echo "Metadados gravados em: $OUTPUT_FILE"
