#!/usr/bin/env bash
# ============================================================================
# P1 Laplace-QPEPLayer 快速复现 harness
#
#   合成 6D 实验（code/run_main.py, code/run_ablation.py）
# + theory 1D 实验（theory/run_unequal.py, run_uniform.py, run_fold.py,
#                  verify_q6.py）
# + 可选 sdp 组（sdp/run_6point_check.py，6 点物理反例，约 1.5 s）
# 默认不跑 real/linemod（需要 BOP LINEMOD 数据；run_symmetry.py 仍是已诊断的负结果，
# run_two_mode.py 是新的真实双峰实验，也需该数据）；
# 加 --with-linemod 可追加。
# 默认不跑 sdp/scorer 组；加 --with-sdp 追加 6 点检查/证书/Q2 图，加 --with-scorer 追加
# P1-Q5c 的 1D+6D Scorer 延拓（~21 s）。
# + 可选 phase-4 真实数据/根计数组（默认都不跑）：
#   --with-roots        40 根快速计数（roots/run_rootcount_fast.py，WM python，~25 s；
#                       期望 results/rootcount_fast.json，配套 docs roots/results/rootcount_fast.md）
#   --with-pose-error   真实 LINEMOD ADD-S/位姿误差代理（run_pose_error.py，WM python，~7 s；
#                       期望 real/linemod/results/pose_error.{json,md,png}）
#   --with-classical    经典 PnP 基线（run_classical_baselines.py，需 sdpr env 的 cv2，
#                       PYTHON_SDPR，~4 s；期望 real/linemod/results/classical_baselines.{json,md}）
#   --with-theseus      Theseus LM/GN 基线（run_theseus_baseline.py，需 theseus venv，
#                       PYTHON_THESEUS，~11 s；还要求同仓库
#                       labs/03-robust-qpeplayer/experiments/baselines；期望
#                       real/linemod/results/theseus_baseline.{json,md}）
#   后三组都需要 BOP LINEMOD 数据与已提交的 real/linemod/results/two_mode.json。
# + 可选 phase-5 终稿组（默认都不跑；代码见 experiments/<path>，数字见 REPRO.md §3.1）：
#   --with-official-sdpr    官方 SDPRLayers 前向基线（baselines/run_official_sdpr.py，
#                           sdpr env，~7 s；要求 third_party/sdprlayer@cb3aed2 与
#                           /home/syk/labs/SDPR/code/sdprlayer@62d8ac6；输出
#                           results/official_sdpr.{json,md}）
#   --with-official-bpnp    官方 BPnP 梯度基线（baselines/run_official_bpnp.py，sdpr env
#                           的 torch+cv2+kornia，~60 s；要求 third_party/BPnP@f1f4f6a、
#                           BOP LINEMOD；输出 results/official_bpnp.{json,md}）
#   --with-feature-frontend 经典预计算特征前端（real/linemod/run_feature_frontend.py，
#                           sdpr env 的 cv2，~35 s；要求 BOP LINEMOD 与已提交的
#                           real/linemod/results/detector_noise.json；输出
#                           real/linemod/results/feature_frontend.{json,md,png}）
#   --with-structured-noise 结构化噪声研究（run_structured_noise.py，sdpr env 的 cv2，
#                           ~2-4 min；同一 detector_noise.json 参考；输出
#                           real/linemod/results/structured_noise.{json,md}）
#   --with-roots-many       40 根计数扩到 11 个实例（roots/run_rootcount_many.py，WM
#                           python，~80 s；输出 results/rootcount_many.json + 配套
#                           roots/results/rootcount_many.md）
#   --with-e2e              端到端 toy 训练/梯度方差/吞吐（code/run_e2e_fit.py，WM python，
#                           完整 20 seeds x 150 steps 约 20-25 min；输出
#                           results/e2e_fit.{json,md,png}）。慢组，默认 off。
#   --with-e2e-smoke        同上但 --smoke（4 seeds x 15 steps，~45 s），输出
#                           results/_smoke_e2e.{json,md,png}；只用于验证接线
#   --with-lmo-full         LMO 全序列两模 + SQPNP（real/linemod/run_two_mode_lmo_full.py，
#                           WM python + sdpr 子进程，177 帧，实测 138 s @ Slurm cpu）。
#                           **重组**：按实验室规范用 srun 提交，见 REPRO.md §3.2；
#                           输出 real/linemod/results/lmo_full.{json,md,png}
#
# 用法（在任意 cwd 下都可执行）：
#   labs/01-laplace-qpeplayer/experiments/run_all.sh
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --with-sdp
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --with-scorer
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --with-linemod
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --with-roots
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --with-pose-error
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --with-classical
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --with-theseus
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --with-sdp-upstream
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --with-official-sdpr
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --with-official-bpnp
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --with-feature-frontend
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --with-structured-noise
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --with-roots-many
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --with-e2e-smoke
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --with-e2e
#   srun --partition=cpu -n1 --time=00:45:00 \
#     labs/01-laplace-qpeplayer/experiments/run_all.sh --with-lmo-full
#   labs/01-laplace-qpeplayer/experiments/run_all.sh --no-compare
#   PYTHON=/path/to/python labs/.../run_all.sh          # 换解释器（默认组）
#   PYTHON_SDPR=/path/to/python labs/.../run_all.sh --with-classical
#   PYTHON_THESEUS=/path/to/python labs/.../run_all.sh --with-theseus
#   P1_STEP_TIMEOUT=1200 labs/.../run_all.sh            # 单脚本超时（秒）
#   P1_E2E_TIMEOUT=3600 labs/.../run_all.sh --with-e2e  # e2e 单独超时（默认 2400 s）
#   P1_LMO_FULL_TIMEOUT=3600 labs/.../run_all.sh --with-lmo-full
#   P1_KEEP_TMP=1 labs/.../run_all.sh                   # 保留临时副本供排查
#   P1_SDPRLAYER=/path/to/sdprlayer labs/.../run_all.sh --with-sdp-upstream
#                                                       # 慢的 upstream 交叉校验
#
# 设计决定（详见 REPRO.md §1）：
#   * 不 in-place 覆盖：把 experiments/ 整棵树复制到 mktemp 目录（按 labs 布局镜像，
#     便于 theseus 组按相对路径 import 03-robust 的 baselines），在副本里跑，
#     已提交的 results/、theory/results/、sdp/results/、roots/results/、
#     real/linemod/results/ 不会被写坏（结束前校验 sha256）。
#   * fresh 输出（主表 JSON/MD/PNG、6 点表、真实数据表等）连同日志一起放到
#     results/logs/<timestamp>/outputs/，方便对照，但不污染已提交结果。
#   * 每个脚本都在一个空 cwd 下执行（clean-cwd 检查）；输出必须落回副本的
#     results/ 目录，否则记为失败。
#   * 默认把 fresh 数值与已提交结果比对（忽略计时字段 t_* / *_ms / runtime），
#     数值漂移 -> 非零退出；--no-compare 可关闭。phase-4 的三张真实数据表
#     （pose_error / classical_baselines / theseus_baseline）只查输出存在，不做
#     数值比对：当前已提交快照由更早的脚本版本生成（字段集/ADD-S 定义不同）。
#
# 退出码：0 = 全部脚本成功 + 输出齐全 + cwd 干净 + 数值与已提交结果一致；
#         1 = 任一脚本失败 / 输出缺失 / cwd 被写 / 数值漂移 / 已提交结果被改动 /
#             phase-4 组缺少解释器、数据或依赖。
# ============================================================================
set -uo pipefail

EXP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-/home/syk/miniconda3/envs/WM/bin/python}
PYTHON_SDPR=${PYTHON_SDPR:-/home/syk/miniconda3/envs/sdpr/bin/python}
PYTHON_THESEUS=${PYTHON_THESEUS:-/mnt/storage1/syk/venvs/theseus/bin/python}
STEP_TIMEOUT=${P1_STEP_TIMEOUT:-600}
# phase-5 long groups get their own (generous) per-step budgets; a one-click run
# must not die on the shared 600 s default half-way through
E2E_TIMEOUT=${P1_E2E_TIMEOUT:-2400}
LMO_FULL_TIMEOUT=${P1_LMO_FULL_TIMEOUT:-1800}
E2E_SMOKE=${P1_E2E_SMOKE:-0}
E2E_PREFIX=e2e_fit
TS=$(date +%Y%m%d-%H%M%S)
LOG_DIR="$EXP_DIR/results/logs/$TS"
RUN_LOG="$LOG_DIR/run.log"
TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/p1-repro.${TS}.XXXXXX")
# mirror the <labs>/<track>/experiments layout, so scripts that resolve sibling
# lab directories via Path(__file__).parents[3] work inside the throwaway copy
LABS_ROOT="$TMP_ROOT/labs"
WORK="$LABS_ROOT/01-laplace-qpeplayer/experiments"
P3_BASELINES_SRC="$EXP_DIR/../../03-robust-qpeplayer/experiments/baselines"
# phase-5 external checkouts / data (hard-coded inside the scripts themselves;
# the harness only preflights them so a missing checkout fails fast)
BOP_LM_TEST=/mnt/storage1/syk/data/bop/lm/test
BOP_LMO_TEST=/mnt/storage1/syk/data/bop/lmo/test/000002
OFFICIAL_SDPR_SRC=/mnt/storage1/syk/third_party/sdprlayer
OFFICIAL_SDPR_FORK=/home/syk/labs/SDPR/code/sdprlayer
BPNP_SRC=/mnt/storage1/syk/third_party/BPnP
CLEAN_CWD="$TMP_ROOT/clean_cwd"
WITH_LINEMOD=0
WITH_SDP=0
WITH_SDP_UPSTREAM=0
WITH_SCORER=0
WITH_ROOTS=0
WITH_POSE_ERROR=0
WITH_CLASSICAL=0
WITH_THESEUS=0
WITH_OFFICIAL_SDPR=0
WITH_OFFICIAL_BPNP=0
WITH_FEATURE_FRONTEND=0
WITH_STRUCTURED_NOISE=0
WITH_ROOTS_MANY=0
WITH_E2E=0
WITH_LMO_FULL=0
DO_COMPARE=1
FAIL=0

usage() {
  # print the leading comment block only (stop at the first non-comment line)
  awk 'NR > 1 { if ($0 ~ /^#/) { sub(/^# ?/, ""); print } else { exit } }' "${BASH_SOURCE[0]}"
}

for arg in "$@"; do
  case "$arg" in
    --with-linemod)     WITH_LINEMOD=1 ;;
    --with-sdp)         WITH_SDP=1 ;;
    --with-scorer)      WITH_SCORER=1 ;;
    --with-roots)       WITH_ROOTS=1 ;;
    --with-pose-error)  WITH_POSE_ERROR=1 ;;
    --with-classical)   WITH_CLASSICAL=1 ;;
    --with-theseus)     WITH_THESEUS=1 ;;
    --with-sdp-upstream) WITH_SDP=1; WITH_SDP_UPSTREAM=1 ;;
    --with-official-sdpr)   WITH_OFFICIAL_SDPR=1 ;;
    --with-official-bpnp)   WITH_OFFICIAL_BPNP=1 ;;
    --with-feature-frontend) WITH_FEATURE_FRONTEND=1 ;;
    --with-structured-noise) WITH_STRUCTURED_NOISE=1 ;;
    --with-roots-many)      WITH_ROOTS_MANY=1 ;;
    --with-e2e)             WITH_E2E=1 ;;
    --with-e2e-smoke)       WITH_E2E=1; E2E_SMOKE=1 ;;
    --with-lmo-full)        WITH_LMO_FULL=1 ;;
    --no-compare)       DO_COMPARE=0 ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; usage; exit 2 ;;
  esac
done

# the smoke variant writes a different prefix so a full run's committed
# results/e2e_fit.* is never mistaken for a 4-seed smoke output
if [[ "$E2E_SMOKE" == "1" ]]; then
  E2E_PREFIX=_smoke_e2e
fi

say() { printf '%s\n' "$*" >&2; printf '%s\n' "$*" >>"$RUN_LOG"; }

cleanup() {
  if [[ "${P1_KEEP_TMP:-0}" == "1" ]]; then
    echo "[run_all] kept temp dir: $TMP_ROOT" >&2
  else
    rm -rf -- "$TMP_ROOT"
  fi
}
trap cleanup EXIT

mkdir -p "$LOG_DIR"

# ---- preflight -------------------------------------------------------------
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: python not executable: $PYTHON" >&2
  exit 1
fi
if ! "$PYTHON" -c 'import numpy, scipy' >/dev/null 2>&1; then
  echo "ERROR: $PYTHON cannot import numpy/scipy" >&2
  exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "ERROR: GNU timeout not found (needed for per-step timeout)" >&2
  exit 1
fi
if [[ "$WITH_SDP" == "1" && ! -f "$EXP_DIR/sdp/run_6point_check.py" ]]; then
  echo "ERROR: --with-sdp requires $EXP_DIR/sdp/run_6point_check.py" >&2
  exit 1
fi
if [[ "$WITH_SCORER" == "1" && ! -f "$EXP_DIR/theory/run_scorer_fold.py" ]]; then
  echo "ERROR: --with-scorer requires $EXP_DIR/theory/run_scorer_fold.py" >&2
  exit 1
fi
if [[ "$WITH_SDP_UPSTREAM" == "1" ]]; then
  if [[ -z "${P1_SDPRLAYER:-}" || ! -d "${P1_SDPRLAYER:-}" ]]; then
    echo "ERROR: --with-sdp-upstream requires P1_SDPRLAYER=/path/to/sdprlayer (an existing dir)" >&2
    exit 1
  fi
fi
# ---- phase-4 optional groups ----------------------------------------------
if [[ "$WITH_ROOTS" == "1" && ! -f "$EXP_DIR/roots/run_rootcount_fast.py" ]]; then
  echo "ERROR: --with-roots requires $EXP_DIR/roots/run_rootcount_fast.py" >&2
  exit 1
fi
if [[ "$WITH_POSE_ERROR" == "1" && ! -f "$EXP_DIR/real/linemod/run_pose_error.py" ]]; then
  echo "ERROR: --with-pose-error requires $EXP_DIR/real/linemod/run_pose_error.py" >&2
  exit 1
fi
if [[ "$WITH_CLASSICAL" == "1" ]]; then
  if [[ ! -f "$EXP_DIR/real/linemod/run_classical_baselines.py" ]]; then
    echo "ERROR: --with-classical requires $EXP_DIR/real/linemod/run_classical_baselines.py" >&2
    exit 1
  fi
  if [[ ! -x "$PYTHON_SDPR" ]]; then
    echo "ERROR: --with-classical requires the sdpr env python at PYTHON_SDPR=$PYTHON_SDPR (missing/not executable); set PYTHON_SDPR=/path/to/python" >&2
    exit 1
  fi
  if ! "$PYTHON_SDPR" -c 'import cv2, PIL, scipy' >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_SDPR cannot import cv2/PIL/scipy (needed by --with-classical)" >&2
    exit 1
  fi
fi
if [[ "$WITH_THESEUS" == "1" ]]; then
  if [[ ! -f "$EXP_DIR/real/linemod/run_theseus_baseline.py" ]]; then
    echo "ERROR: --with-theseus requires $EXP_DIR/real/linemod/run_theseus_baseline.py" >&2
    exit 1
  fi
  if [[ ! -x "$PYTHON_THESEUS" ]]; then
    echo "ERROR: --with-theseus requires the theseus venv at PYTHON_THESEUS=$PYTHON_THESEUS (missing/not executable); set PYTHON_THESEUS=/path/to/python or omit --with-theseus" >&2
    exit 1
  fi
  if ! "$PYTHON_THESEUS" -c 'import theseus, torch' >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_THESEUS cannot import theseus/torch (needed by --with-theseus)" >&2
    exit 1
  fi
  if [[ ! -d "$P3_BASELINES_SRC" ]]; then
    echo "ERROR: --with-theseus requires the P3 baselines at $P3_BASELINES_SRC (theseus_cost.py is imported)" >&2
    exit 1
  fi
fi
if [[ "$WITH_POSE_ERROR" == "1" || "$WITH_CLASSICAL" == "1" || "$WITH_THESEUS" == "1" ]]; then
  if [[ ! -d /mnt/storage1/syk/data/bop/lm/test ]]; then
    echo "ERROR: --with-pose-error/--with-classical/--with-theseus require BOP LINEMOD at /mnt/storage1/syk/data/bop/lm/test" >&2
    exit 1
  fi
  if [[ ! -f "$EXP_DIR/real/linemod/results/two_mode.json" ]]; then
    echo "ERROR: the real-data groups require the committed reference $EXP_DIR/real/linemod/results/two_mode.json" >&2
    exit 1
  fi
fi

# ---- phase-5 optional groups (final push) ---------------------------------
need_sdpr_python() {  # $1 = group name, $2 = modules to import
  if [[ ! -x "$PYTHON_SDPR" ]]; then
    echo "ERROR: $1 requires the sdpr env python at PYTHON_SDPR=$PYTHON_SDPR (missing/not executable); set PYTHON_SDPR=/path/to/python or omit $1" >&2
    exit 1
  fi
  if ! "$PYTHON_SDPR" -c "import $2" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_SDPR cannot import $2 (needed by $1)" >&2
    exit 1
  fi
}
if [[ "$WITH_OFFICIAL_SDPR" == "1" ]]; then
  if [[ ! -f "$EXP_DIR/baselines/run_official_sdpr.py" ]]; then
    echo "ERROR: --with-official-sdpr requires $EXP_DIR/baselines/run_official_sdpr.py" >&2
    exit 1
  fi
  need_sdpr_python --with-official-sdpr 'cvxpy, torch, sdprlayers'
  if [[ ! -d "$OFFICIAL_SDPR_SRC/.git" ]]; then
    echo "ERROR: --with-official-sdpr requires the released tree at $OFFICIAL_SDPR_SRC (git repo @ cb3aed2, exported with git archive)" >&2
    exit 1
  fi
  if [[ ! -d "$OFFICIAL_SDPR_FORK" ]]; then
    echo "ERROR: --with-official-sdpr requires the local sdprlayer fork at $OFFICIAL_SDPR_FORK (checked as the second implementation)" >&2
    exit 1
  fi
fi
if [[ "$WITH_OFFICIAL_BPNP" == "1" ]]; then
  if [[ ! -f "$EXP_DIR/baselines/run_official_bpnp.py" ]]; then
    echo "ERROR: --with-official-bpnp requires $EXP_DIR/baselines/run_official_bpnp.py" >&2
    exit 1
  fi
  need_sdpr_python --with-official-bpnp 'torch, cv2, kornia'
  if [[ ! -f "$BPNP_SRC/BPnP.py" ]]; then
    echo "ERROR: --with-official-bpnp requires the released BPnP at $BPNP_SRC (BPnP.py)" >&2
    exit 1
  fi
  if [[ ! -d "$BOP_LM_TEST" ]]; then
    echo "ERROR: --with-official-bpnp requires BOP LINEMOD at $BOP_LM_TEST (2 real frames)" >&2
    exit 1
  fi
fi
if [[ "$WITH_FEATURE_FRONTEND" == "1" || "$WITH_STRUCTURED_NOISE" == "1" ]]; then
  grp=--with-feature-frontend; script=run_feature_frontend.py
  [[ "$WITH_STRUCTURED_NOISE" == "1" ]] && { grp=--with-structured-noise; script=run_structured_noise.py; }
  if [[ ! -f "$EXP_DIR/real/linemod/$script" ]]; then
    echo "ERROR: $grp requires $EXP_DIR/real/linemod/$script" >&2
    exit 1
  fi
  need_sdpr_python "$grp" 'cv2, PIL, scipy'
  if [[ ! -d "$BOP_LM_TEST" ]]; then
    echo "ERROR: $grp requires BOP LINEMOD at $BOP_LM_TEST" >&2
    exit 1
  fi
  if [[ ! -f "$EXP_DIR/real/linemod/results/detector_noise.json" ]]; then
    echo "ERROR: $grp requires the committed reference $EXP_DIR/real/linemod/results/detector_noise.json (frame set + clean baseline)" >&2
    exit 1
  fi
fi
if [[ "$WITH_ROOTS_MANY" == "1" ]]; then
  if [[ ! -f "$EXP_DIR/roots/run_rootcount_many.py" ]]; then
    echo "ERROR: --with-roots-many requires $EXP_DIR/roots/run_rootcount_many.py" >&2
    exit 1
  fi
  if [[ ! -f "$EXP_DIR/roots/run_rootcount_fast.py" ]]; then
    echo "ERROR: --with-roots-many requires $EXP_DIR/roots/run_rootcount_fast.py (imported as a module)" >&2
    exit 1
  fi
fi
if [[ "$WITH_E2E" == "1" ]]; then
  if [[ ! -f "$EXP_DIR/code/run_e2e_fit.py" ]]; then
    echo "ERROR: --with-e2e/--with-e2e-smoke requires $EXP_DIR/code/run_e2e_fit.py" >&2
    exit 1
  fi
fi
if [[ "$WITH_LMO_FULL" == "1" ]]; then
  if [[ ! -f "$EXP_DIR/real/linemod/run_two_mode_lmo_full.py" ]]; then
    echo "ERROR: --with-lmo-full requires $EXP_DIR/real/linemod/run_two_mode_lmo_full.py" >&2
    exit 1
  fi
  if [[ ! -f "$EXP_DIR/real/linemod/sqpnp_lmo_helper.py" ]]; then
    echo "ERROR: --with-lmo-full requires $EXP_DIR/real/linemod/sqpnp_lmo_helper.py (SQPNP runs in the sdpr env)" >&2
    exit 1
  fi
  if [[ ! -d "$BOP_LMO_TEST" ]]; then
    echo "ERROR: --with-lmo-full requires the BOP LMO test sequence at $BOP_LMO_TEST (1214 frames)" >&2
    exit 1
  fi
  if [[ ! -x "$PYTHON_SDPR" ]] || ! "$PYTHON_SDPR" -c 'import cv2' >/dev/null 2>&1; then
    echo "ERROR: --with-lmo-full needs cv2 (SQPNP helper) in PYTHON_SDPR=$PYTHON_SDPR" >&2
    exit 1
  fi
  if [[ ! -f "$EXP_DIR/real/linemod/results/lmo_two_mode.json" ]]; then
    echo "WARN: --with-lmo-full will skip the 37-frame stability cross-check ($EXP_DIR/real/linemod/results/lmo_two_mode.json missing)" >&2
  fi
  if [[ -z "${SLURM_JOB_ID:-}" && "${P1_ALLOW_LOGIN_NODE:-0}" != "1" ]]; then
    echo "WARN: --with-lmo-full is a heavy group (177 frames, ~2.5 min on a quiet cpu node, more under load)." >&2
    echo "      Lab policy: submit through Slurm, e.g." >&2
    echo "      srun --partition=cpu -n1 --time=00:45:00 $0 --with-lmo-full" >&2
    echo "      (set P1_ALLOW_LOGIN_NODE=1 to silence this warning)" >&2
  fi
fi

# ---- environment snapshot --------------------------------------------------
{
  echo "timestamp: $(date -Is)"
  echo "host: $(hostname 2>/dev/null || echo n/a)"
  echo "invoked-from-cwd: $PWD"
  echo "python: $PYTHON"
  "$PYTHON" -V
  "$PYTHON" - <<'PY'
import platform
import numpy, scipy, matplotlib, cvxpy
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("matplotlib", matplotlib.__version__)
print("cvxpy", cvxpy.__version__)
print("platform", platform.platform())
PY
  echo "git-head: $(git -C "$EXP_DIR" rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "git-branch: $(git -C "$EXP_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
  echo "sdp-upstream-path: ${P1_SDPRLAYER:-<unset>}"
  if [[ "$WITH_CLASSICAL" == "1" || "$WITH_OFFICIAL_SDPR" == "1" || \
        "$WITH_OFFICIAL_BPNP" == "1" || "$WITH_FEATURE_FRONTEND" == "1" || \
        "$WITH_STRUCTURED_NOISE" == "1" || "$WITH_LMO_FULL" == "1" ]]; then
    echo "python-sdpr: $PYTHON_SDPR"
    "$PYTHON_SDPR" - <<'PY'
import numpy, scipy
print("numpy", numpy.__version__, "| scipy", scipy.__version__)
for name in ("cv2", "torch", "kornia", "cvxpy", "sdprlayers"):
    try:
        m = __import__(name)
        print(name, getattr(m, "__version__", "ok"))
    except Exception as e:  # noqa: BLE001 - report, never abort the snapshot
        print(name, "MISSING", type(e).__name__)
PY
  fi
  if [[ "$WITH_OFFICIAL_SDPR" == "1" ]]; then
    echo "official-sdpr-released: $(git -C "$OFFICIAL_SDPR_SRC" rev-parse --short HEAD 2>/dev/null || echo n/a) ($OFFICIAL_SDPR_SRC)"
    echo "official-sdpr-fork: $(git -C "$OFFICIAL_SDPR_FORK" rev-parse --short HEAD 2>/dev/null || echo n/a) ($OFFICIAL_SDPR_FORK)"
  fi
  if [[ "$WITH_OFFICIAL_BPNP" == "1" ]]; then
    echo "official-bpnp: $(git -C "$BPNP_SRC" rev-parse --short HEAD 2>/dev/null || echo n/a) ($BPNP_SRC)"
  fi
  if [[ "$WITH_LMO_FULL" == "1" ]]; then
    echo "lmo-test: $BOP_LMO_TEST"
    echo "slurm-job-id: ${SLURM_JOB_ID:-<none>}"
  fi
  if [[ "$WITH_THESEUS" == "1" ]]; then
    echo "python-theseus: $PYTHON_THESEUS"
    "$PYTHON_THESEUS" - <<'PY'
import os, theseus, torch
print("theseus", theseus.__version__, "| torch", torch.__version__)
PY
  fi
} >"$LOG_DIR/env.txt" 2>&1

# ---- work in a throwaway copy so committed results are never touched -------
mkdir -p "$WORK"
tar -C "$EXP_DIR" --exclude='./results/logs' -cf - . | tar -C "$WORK" -xf -
# the theseus group imports labs/03-robust-qpeplayer/experiments/baselines/theseus_cost.py
# through a path relative to the copied tree; mirror that sibling lab in the copy.
if [[ "$WITH_THESEUS" == "1" ]]; then
  mkdir -p "$LABS_ROOT/03-robust-qpeplayer/experiments"
  cp -a "$P3_BASELINES_SRC" "$LABS_ROOT/03-robust-qpeplayer/experiments/baselines"
fi
mkdir -p "$CLEAN_CWD" "$TMP_ROOT/baseline" "$TMP_ROOT/baseline/theory"
cp -a "$WORK/results" "$TMP_ROOT/baseline/results"
cp -a "$WORK/theory/results" "$TMP_ROOT/baseline/theory/results"
if [[ -d "$WORK/real/linemod/results" ]]; then
  mkdir -p "$TMP_ROOT/baseline/real/linemod"
  cp -a "$WORK/real/linemod/results" "$TMP_ROOT/baseline/real/linemod/results"
fi
if [[ -d "$WORK/sdp/results" ]]; then
  mkdir -p "$TMP_ROOT/baseline/sdp"
  cp -a "$WORK/sdp/results" "$TMP_ROOT/baseline/sdp/results"
fi
if [[ ! -d "$TMP_ROOT/baseline/theory/results" ]]; then
  echo "ERROR: failed to snapshot committed theory results for comparison" >&2
  exit 1
fi

results_manifest() {
  ( cd "$EXP_DIR" && find results theory/results real/linemod/results sdp/results roots/results \
      -type f ! -path 'results/logs/*' -print0 2>/dev/null \
    | sort -z | xargs -0 -r sha256sum 2>/dev/null )
}
BEFORE_MANIFEST="$LOG_DIR/results_before.sha256"
AFTER_MANIFEST="$LOG_DIR/results_after.sha256"
results_manifest >"$BEFORE_MANIFEST"

cat >"$RUN_LOG" <<EOF
# P1 Laplace-QPEPLayer fast-suite reproduction
timestamp : $TS
host      : $(hostname 2>/dev/null || echo n/a)
exp_dir   : $EXP_DIR
python    : $PYTHON
workdir   : $TMP_ROOT (throwaway copy; committed results untouched)
flags     : with_linemod=$WITH_LINEMOD with_sdp=$WITH_SDP with_sdp_upstream=$WITH_SDP_UPSTREAM with_scorer=$WITH_SCORER
            with_roots=$WITH_ROOTS with_pose_error=$WITH_POSE_ERROR with_classical=$WITH_CLASSICAL with_theseus=$WITH_THESEUS
            with_official_sdpr=$WITH_OFFICIAL_SDPR with_official_bpnp=$WITH_OFFICIAL_BPNP with_feature_frontend=$WITH_FEATURE_FRONTEND
            with_structured_noise=$WITH_STRUCTURED_NOISE with_roots_many=$WITH_ROOTS_MANY with_e2e=$WITH_E2E with_lmo_full=$WITH_LMO_FULL
            compare=$DO_COMPARE
python_sdpr  : $PYTHON_SDPR
python_theseus: $PYTHON_THESEUS
step_timeouts: default=${STEP_TIMEOUT}s e2e=${E2E_TIMEOUT}s lmo_full=${LMO_FULL_TIMEOUT}s
EOF

say "== preflight ok: $("$PYTHON" -V 2>&1)"
say "== log dir: $LOG_DIR"
say ""

# ---- per-script expectations ----------------------------------------------
expected_for() {
  case "$(basename "$1")" in
    run_main.py)
      echo "results/main_table.json results/raw_grid.json results/main_table.md results/main_table.png" ;;
    run_ablation.py)
      echo "results/ablation.json results/ablation.md" ;;
    run_unequal.py)
      echo "theory/results/uniform_unequal.json theory/results/uniform_unequal_table.md" ;;
    run_uniform.py)
      echo "theory/results/uniform.json theory/results/uniform_table.md theory/results/deltaJ_sweep.json theory/results/deltaJ_sweep.md" ;;
    run_fold.py)
      echo "theory/results/fold.json theory/results/fold_table.md" ;;
    verify_q6.py)
      echo "theory/results/q6_verify.json theory/results/q6_verify.md" ;;
    run_6point_check.py)
      echo "sdp/results/six_point.json sdp/results/six_point.md" ;;
    run_six_point_cert.py)
      echo "sdp/results/six_point_cert.json sdp/results/six_point_cert.md" ;;
    make_q2_summary.py)
      echo "sdp/results/q2_summary.json sdp/results/q2_summary.png" ;;
    run_sdp_check.py)
      echo "sdp/results/sdp_check.json sdp/results/sdp_check.md" ;;
    run_symmetry.py)
      echo "real/linemod/results/symmetry.json real/linemod/results/symmetry.md" ;;
    run_pose_error.py)
      echo "real/linemod/results/pose_error.json real/linemod/results/pose_error.md real/linemod/results/pose_error.png" ;;
    run_classical_baselines.py)
      echo "real/linemod/results/classical_baselines.json real/linemod/results/classical_baselines.md" ;;
    run_theseus_baseline.py)
      echo "real/linemod/results/theseus_baseline.json real/linemod/results/theseus_baseline.md" ;;
    run_rootcount_fast.py)
      # the script only writes the JSON; roots/results/rootcount_fast.md is a
      # committed human-readable summary checked separately below
      echo "results/rootcount_fast.json" ;;
    run_rootcount_many.py)
      echo "results/rootcount_many.json" ;;
    run_official_sdpr.py)
      echo "results/official_sdpr.json results/official_sdpr.md" ;;
    run_official_bpnp.py)
      echo "results/official_bpnp.json results/official_bpnp.md" ;;
    run_feature_frontend.py)
      echo "real/linemod/results/feature_frontend.json real/linemod/results/feature_frontend.md real/linemod/results/feature_frontend.png" ;;
    run_structured_noise.py)
      echo "real/linemod/results/structured_noise.json real/linemod/results/structured_noise.md" ;;
    run_two_mode_lmo_full.py)
      echo "real/linemod/results/lmo_full.json real/linemod/results/lmo_full.md real/linemod/results/lmo_full.png" ;;
    run_e2e_fit.py)
      # --smoke writes the _smoke_e2e prefix (never overwrites e2e_fit.*)
      echo "results/${E2E_PREFIX}.json results/${E2E_PREFIX}.md results/${E2E_PREFIX}.png" ;;
    *)
      echo "" ;;
  esac
}

SCRIPTS=(
  code/run_main.py
  code/run_ablation.py
  theory/run_unequal.py
  theory/run_uniform.py
  theory/run_fold.py
  theory/verify_q6.py
)
declare -A EXTRA_ARGS=()
# per-script interpreter override (defaults to $PYTHON); phase-4 groups need
# cv2 (sdpr env) and theseus/torch (theseus venv) respectively
declare -A STEP_PYTHON=()
if [[ "$WITH_SDP" == "1" ]]; then
  SCRIPTS+=(sdp/run_6point_check.py sdp/run_six_point_cert.py sdp/make_q2_summary.py)
fi
if [[ "$WITH_SCORER" == "1" ]]; then
  SCRIPTS+=(theory/run_scorer_fold.py)
fi
if [[ "$WITH_SCORER" == "1" && ! -f "$EXP_DIR/theory/run_scorer_fold.py" ]]; then
  echo "ERROR: --with-scorer requires $EXP_DIR/theory/run_scorer_fold.py" >&2
  exit 1
fi
if [[ "$WITH_SDP_UPSTREAM" == "1" ]]; then
  # slow (~1 min, needs an external sdprlayer checkout); never on by default
  SCRIPTS+=(sdp/run_sdp_check.py)
  EXTRA_ARGS[sdp/run_sdp_check.py]="--check-upstream $P1_SDPRLAYER"
fi
# ---- phase-4 optional groups (all off by default; each is its own group) ---
if [[ "$WITH_ROOTS" == "1" ]]; then
  SCRIPTS+=(roots/run_rootcount_fast.py)
fi
if [[ "$WITH_POSE_ERROR" == "1" ]]; then
  SCRIPTS+=(real/linemod/run_pose_error.py)
fi
if [[ "$WITH_CLASSICAL" == "1" ]]; then
  SCRIPTS+=(real/linemod/run_classical_baselines.py)
  STEP_PYTHON[real/linemod/run_classical_baselines.py]="$PYTHON_SDPR"
fi
if [[ "$WITH_THESEUS" == "1" ]]; then
  SCRIPTS+=(real/linemod/run_theseus_baseline.py)
  STEP_PYTHON[real/linemod/run_theseus_baseline.py]="$PYTHON_THESEUS"
fi
if [[ "$WITH_LINEMOD" == "1" ]]; then
  if [[ ! -d /mnt/storage1/syk/data/bop/lm/test ]]; then
    say "ERROR: --with-linemod requires BOP LINEMOD at /mnt/storage1/syk/data/bop/lm"
    exit 1
  fi
  SCRIPTS+=(real/linemod/run_symmetry.py)
fi
# ---- phase-5 optional groups (final push; all off by default) -------------
if [[ "$WITH_OFFICIAL_SDPR" == "1" ]]; then
  SCRIPTS+=(baselines/run_official_sdpr.py)
  STEP_PYTHON[baselines/run_official_sdpr.py]="$PYTHON_SDPR"
fi
if [[ "$WITH_OFFICIAL_BPNP" == "1" ]]; then
  SCRIPTS+=(baselines/run_official_bpnp.py)
  STEP_PYTHON[baselines/run_official_bpnp.py]="$PYTHON_SDPR"
fi
if [[ "$WITH_FEATURE_FRONTEND" == "1" ]]; then
  SCRIPTS+=(real/linemod/run_feature_frontend.py)
  STEP_PYTHON[real/linemod/run_feature_frontend.py]="$PYTHON_SDPR"
fi
if [[ "$WITH_STRUCTURED_NOISE" == "1" ]]; then
  SCRIPTS+=(real/linemod/run_structured_noise.py)
  STEP_PYTHON[real/linemod/run_structured_noise.py]="$PYTHON_SDPR"
fi
if [[ "$WITH_ROOTS_MANY" == "1" ]]; then
  SCRIPTS+=(roots/run_rootcount_many.py)
fi
if [[ "$WITH_E2E" == "1" ]]; then
  SCRIPTS+=(code/run_e2e_fit.py)
  if [[ "$E2E_SMOKE" == "1" ]]; then
    EXTRA_ARGS[code/run_e2e_fit.py]="--smoke"
  fi
fi
if [[ "$WITH_LMO_FULL" == "1" ]]; then
  # WM python (the estimator env); the SQPNP helper is spawned inside the
  # script with the hard-coded sdpr interpreter
  SCRIPTS+=(real/linemod/run_two_mode_lmo_full.py)
fi

# per-step timeout: the shared default would kill e2e / lmo_full mid-way
step_timeout_for() {
  case "$(basename "$1")" in
    run_e2e_fit.py) echo "$E2E_TIMEOUT" ;;
    run_two_mode_lmo_full.py) echo "$LMO_FULL_TIMEOUT" ;;
    *) echo "$STEP_TIMEOUT" ;;
  esac
}

declare -a NAMES RCS MSECS NOTES
RUN_START=$(date +%s%N)

for rel in "${SCRIPTS[@]}"; do
  name=$(basename "$rel")
  out="$LOG_DIR/${name}.stdout"
  err="$LOG_DIR/${name}.stderr"
  # remove stale copies of this script's outputs first, so a crashed script can
  # never "pass" by leaving the committed numbers in place
  for f in $(expected_for "$rel"); do rm -f -- "$WORK/$f"; done
  extra=()
  if [[ -n "${EXTRA_ARGS[$rel]:-}" ]]; then
    read -r -a extra <<<"${EXTRA_ARGS[$rel]}"
  fi
  t0=$(date +%s%N)
  py=${STEP_PYTHON[$rel]:-$PYTHON}
  step_to=$(step_timeout_for "$rel")
  ( cd "$CLEAN_CWD" && timeout "${step_to}s" "$py" "$WORK/$rel" "${extra[@]}" ) >"$out" 2>"$err"
  rc=$?
  t1=$(date +%s%N)
  ms=$(( (t1 - t0) / 1000000 ))

  note=""
  # 1) expected outputs present and non-empty, under the copy's results dirs
  missing=()
  for f in $(expected_for "$rel"); do
    [[ -s "$WORK/$f" ]] || missing+=("$f")
  done
  if (( ${#missing[@]} > 0 )); then
    note+="missing: ${missing[*]}; "
  fi
  # 2) nothing may be written into the clean cwd
  if [[ -n "$(find "$CLEAN_CWD" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
    note+="cwd polluted: $(find "$CLEAN_CWD" -mindepth 1 -maxdepth 1 -printf '%f ' 2>/dev/null); "
  fi
  # 3) timeout 124 / other non-zero exit
  if [[ "$rc" != "0" ]]; then
    note+="rc=$rc; "
    tail -n 25 "$err" >>"$RUN_LOG" 2>/dev/null || true
  fi
  # 4) known negative result: linemod currently emits an empty table (see REPRO.md)
  if [[ "$name" == "run_symmetry.py" && -f "$WORK/real/linemod/results/symmetry.json" ]]; then
    if [[ "$(tr -d '[:space:]' <"$WORK/real/linemod/results/symmetry.json")" == "[]" ]]; then
      note+="negative: 0 rows (known, REPRO.md B1); "
    fi
  fi

  NAMES+=("$rel"); RCS+=("$rc"); MSECS+=("$ms"); NOTES+=("$note")
  if [[ "$rc" != "0" || ${#missing[@]} -gt 0 ]]; then
    FAIL=1
  fi
  printf -v secs '%d.%03d' $((ms / 1000)) $((ms % 1000))
  say "[$(printf '%-22s' "$name")] rc=$rc  wall=${secs}s  ${note}"
done

# ---- companion docs that the scripts do not regenerate ---------------------
# roots/run_rootcount_fast.py only writes results/rootcount_fast.json; the
# human-readable roots/results/rootcount_fast.md is a committed summary.
if [[ "$WITH_ROOTS" == "1" ]]; then
  if [[ -s "$EXP_DIR/roots/results/rootcount_fast.md" ]]; then
    say "== docs check ok: roots/results/rootcount_fast.md present (not regenerated by the script)"
  else
    say "!! --with-roots expects the committed summary roots/results/rootcount_fast.md (the script only writes the JSON)"
    FAIL=1
  fi
fi
if [[ "$WITH_ROOTS_MANY" == "1" ]]; then
  if [[ -s "$EXP_DIR/roots/results/rootcount_many.md" ]]; then
    say "== docs check ok: roots/results/rootcount_many.md present (not regenerated by the script)"
  else
    say "!! --with-roots-many expects the committed summary roots/results/rootcount_many.md (the script only writes the JSON)"
    FAIL=1
  fi
fi

RUN_END=$(date +%s%N)
TOTAL_MS=$(( (RUN_END - RUN_START) / 1000000 ))

# ---- compare fresh numbers with committed results (timing fields ignored) --
if [[ "$DO_COMPARE" == "1" ]]; then
  PAIRS=(
    results/main_table.json
    results/raw_grid.json
    results/ablation.json
    theory/results/uniform_unequal.json
    theory/results/uniform.json
    theory/results/deltaJ_sweep.json
    theory/results/fold.json
    theory/results/q6_verify.json
  )
  if [[ "$WITH_LINEMOD" == "1" ]]; then
    PAIRS+=(real/linemod/results/symmetry.json)
  fi
  if [[ "$WITH_SDP" == "1" ]]; then
    PAIRS+=(sdp/results/six_point.json)
  fi
  if [[ "$WITH_SCORER" == "1" ]]; then
    PAIRS+=(theory/results/scorer_fold.json)
  fi
  if [[ "$WITH_ROOTS" == "1" ]]; then
    # root counts / residuals are deterministic; only per-instance runtime differs
    PAIRS+=(results/rootcount_fast.json)
  fi
  if [[ "$WITH_ROOTS_MANY" == "1" ]]; then
    PAIRS+=(results/rootcount_many.json)
  fi
  if [[ "$WITH_E2E" == "1" && "$E2E_SMOKE" == "1" ]]; then
    # the 4-seed smoke row is fully seeded; wall_seconds is stripped below
    PAIRS+=("results/${E2E_PREFIX}.json")
  fi
  # --with-pose-error / --with-classical / --with-theseus deliberately do NOT
  # add their JSONs to PAIRS: the committed snapshots were produced by earlier
  # script revisions (pose_error ADD-S target set changed to the dense GT mesh;
  # classical/theseus gained new fields), so a numeric compare would flag the
  # revision difference, not a reproducibility failure. Their outputs are still
  # checked for existence + non-emptiness above and archived in outputs/.
  # the slow --with-sdp-upstream run is a smoke test only: outputs must exist,
  # but sdp_check.json is not compared numerically (solver-dependent fields).
  "$PYTHON" - "$WORK" "$TMP_ROOT/baseline" "${PAIRS[@]}" <<'PY' | tee -a "$RUN_LOG"
import json, math, sys
from pathlib import Path

work, base = Path(sys.argv[1]), Path(sys.argv[2])
files = sys.argv[3:]

def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items()
                if not (isinstance(k, str) and
                        (k.startswith("t_") or k.endswith("_ms") or
                         k in ("runtime", "runtime_s", "wall_time_s",
                               "wall_seconds", "wall_s", "wall_s_total")))}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj

def walk(a, b, path, out):
    if type(a) is not type(b):
        out.append((path, "type", type(a).__name__, type(b).__name__)); return
    if isinstance(a, dict):
        ka, kb = set(a), set(b)
        if ka != kb:
            out.append((path, "keys", sorted(ka - kb), sorted(kb - ka))); return
        for k in sorted(ka):
            walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append((path, "len", len(a), len(b))); return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float) or isinstance(b, float):
        x, y = float(a), float(b)
        if math.isnan(x) and math.isnan(y):
            return
        d = abs(x - y)
        if d > 1e-15 and d > 1e-9 * max(abs(x), abs(y)):
            out.append((path, "float", repr(a), repr(b), d))
    elif a != b:
        out.append((path, "value", repr(a), repr(b)))

bad = 0
for rel in files:
    pa, pb = work / rel, base / rel
    if not pa.exists():
        print(f"COMPARE {rel}: MISSING fresh output"); bad += 1; continue
    if not pb.exists():
        print(f"COMPARE {rel}: no committed baseline (skipped)"); continue
    a, b = strip(json.loads(pa.read_text())), strip(json.loads(pb.read_text()))
    diffs = []
    walk(a, b, "", diffs)
    if diffs:
        bad += 1
        print(f"COMPARE {rel}: {len(diffs)} diff(s) in non-timing fields vs committed")
        for d in diffs[:5]:
            print("   ", d)
    else:
        print(f"COMPARE {rel}: OK (non-timing fields match committed)")
sys.exit(1 if bad else 0)
PY
  cmp_rc=$?
  if [[ "$cmp_rc" != "0" ]]; then
    FAIL=1
    say "!! numeric comparison FAILED (see COMPARE lines above)"
  fi
fi

# ---- committed results must be untouched ----------------------------------
results_manifest >"$AFTER_MANIFEST"
if ! cmp -s "$BEFORE_MANIFEST" "$AFTER_MANIFEST"; then
  FAIL=1
  say "!! committed results changed while running the harness:"
  diff "$BEFORE_MANIFEST" "$AFTER_MANIFEST" | sed -n '1,20p' >>"$RUN_LOG"
fi

# ---- archive fresh outputs + hashes ---------------------------------------
mkdir -p "$LOG_DIR/outputs"
cp -a "$WORK/results" "$LOG_DIR/outputs/results"
cp -a "$WORK/theory/results" "$LOG_DIR/outputs/theory_results"
if [[ "$WITH_LINEMOD$WITH_POSE_ERROR$WITH_CLASSICAL$WITH_THESEUS$WITH_FEATURE_FRONTEND$WITH_STRUCTURED_NOISE$WITH_LMO_FULL" == *1* && -d "$WORK/real/linemod/results" ]]; then
  cp -a "$WORK/real/linemod/results" "$LOG_DIR/outputs/linemod_results"
fi
if [[ "$WITH_SDP" == "1" && -d "$WORK/sdp/results" ]]; then
  cp -a "$WORK/sdp/results" "$LOG_DIR/outputs/sdp_results"
fi
# rootcount_fast.json lands in results/, already archived above; keep the
# committed roots/results/rootcount_fast.md snapshot in the archived tree too
if [[ "$WITH_ROOTS$WITH_ROOTS_MANY" == *1* && -d "$WORK/roots/results" ]]; then
  cp -a "$WORK/roots/results" "$LOG_DIR/outputs/roots_results"
fi
( cd "$LOG_DIR/outputs" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum ) \
  >"$LOG_DIR/outputs.sha256"

# ---- summary ---------------------------------------------------------------
printf -v total_secs '%d.%03d' $((TOTAL_MS / 1000)) $((TOTAL_MS % 1000))
{
  echo ""
  echo "## summary ($TS)"
  echo "| script | rc | wall (s) | note |"
  echo "|-|-|-|-|"
  for i in "${!NAMES[@]}"; do
    printf -v s '%d.%03d' $((MSECS[i] / 1000)) $((MSECS[i] % 1000))
    echo "| ${NAMES[i]} | ${RCS[i]} | ${s} | ${NOTES[i]:-ok} |"
  done
  echo ""
  echo "total wall: ${total_secs}s | env: env.txt | fresh outputs: outputs/ | hashes: outputs.sha256"
} | tee -a "$RUN_LOG" >&2

if [[ "$FAIL" == "0" ]]; then
  say "RESULT: PASS (log: $RUN_LOG)"
  exit 0
else
  say "RESULT: FAIL (log: $RUN_LOG)"
  exit 1
fi
