#!/usr/bin/env bash
set -euo pipefail

ERRORS=0

_check() {
    local desc="$1"
    shift
    if "$@" &>/dev/null; then
        echo "[OK]   $desc"
    else
        echo "[FAIL] $desc"
        ERRORS=$((ERRORS + 1))
    fi
}

echo "=== Verificando ambiente ==="
echo ""

_check "nvidia-smi disponível e funcional" nvidia-smi
_check "python3 disponível" command -v python3
_check "CuPy importa sem erro" python3 -c "import cupy; print(cupy.__version__)"
_check "CUDA acessível via CuPy (getDeviceCount)" \
    python3 -c "import cupy; cupy.cuda.runtime.getDeviceCount()"

echo ""
echo "=== Versões detectadas ==="

if command -v nvidia-smi &>/dev/null; then
    echo "Driver NVIDIA : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "CUDA (runtime): $(python3 -c 'import cupy; v=cupy.cuda.runtime.runtimeGetVersion(); print(f\"{v//1000}.{(v%1000)//10}\")'  2>/dev/null || echo 'N/A')"
    echo "GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
else
    echo "(nvidia-smi indisponível)"
fi

if command -v python3 &>/dev/null; then
    echo "Python        : $(python3 --version 2>&1 || echo 'N/A')"
    echo "CuPy          : $(python3 -c 'import cupy; print(cupy.__version__)' 2>/dev/null || echo 'N/A')"
fi

echo ""
if [[ $ERRORS -gt 0 ]]; then
    echo "RESULTADO: FALHA — $ERRORS verificação(ões) com erro."
    exit 1
fi

echo "RESULTADO: OK — ambiente verificado com sucesso."
exit 0
