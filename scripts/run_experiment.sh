#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# a. Verificar ambiente — aborta se falhar
echo "=== [1/4] Verificando ambiente ==="
"$SCRIPT_DIR/verify_environment.sh"

# b. Timestamp único para todos os artefatos desta execução
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo ""
echo "=== Timestamp da execução: $TIMESTAMP ==="

# c. Coletar metadados
echo ""
echo "=== [2/4] Coletando metadados ==="
"$SCRIPT_DIR/collect_metadata.sh" "$TIMESTAMP"

# d. Executar benchmark (sem prompt interativo)
echo ""
echo "=== [3/4] Executando benchmark GEMM ==="
cd "$PROJECT_ROOT"
python3 -m src.benchmark.gemm_benchmark --timestamp "$TIMESTAMP" --no-confirm

# e. Análise dos resultados
echo ""
echo "=== [4/4] Rodando análise ==="
python3 -m src.analysis.run_analysis \
    --csv "data/raw/gemm_rtx5060ti_${TIMESTAMP}.csv"

# f. Resumo dos artefatos gerados
echo ""
echo "=== Artefatos gerados ==="
echo "CSV bruto    : $PROJECT_ROOT/data/raw/gemm_rtx5060ti_${TIMESTAMP}.csv"
echo "Metadados    : $PROJECT_ROOT/data/raw/metadata_${TIMESTAMP}.json"
echo "Figuras      : $PROJECT_ROOT/figures/"
