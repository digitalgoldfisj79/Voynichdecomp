#!/data/data/com.termux/files/usr/bin/bash
# ═══════════════════════════════════════════════════════════════
# termux_reproduce_s3.sh — Run PGCS S3 reproduction on Termux
# ═══════════════════════════════════════════════════════════════
#
# Usage:
#   bash termux_reproduce_s3.sh          # Full setup + run
#   bash termux_reproduce_s3.sh --run    # Skip setup, just run
#   bash termux_reproduce_s3.sh --quick  # Skip BG22 + ablation (fastest)
#
# Estimated runtime:
#   --quick:  15-25 min (template generators only)
#   default:  40-90 min (all 23 generators)
#
# Requirements: ~500MB free storage, ~1GB free RAM
# ═══════════════════════════════════════════════════════════════

set -e

REPO_URL="https://github.com/digitalgoldfisj79/Voynichdecomp.git"
WORK_DIR="$HOME/Voynichdecomp"
PAPER_DIR="$WORK_DIR/Paper"

# ── Parse args ──
SKIP_SETUP=0
QUICK=0
EXTRA_ARGS=""
for arg in "$@"; do
    case $arg in
        --run)     SKIP_SETUP=1 ;;
        --quick)   QUICK=1 ;;
        --resume)  EXTRA_ARGS="$EXTRA_ARGS --resume" ;;
        --force)   EXTRA_ARGS="$EXTRA_ARGS --force" ;;
        *)         EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

# ── Colours for readability on small screens ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[S3]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!!]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# ═══════════════════════════════════════════════════════════════
# SETUP (skip with --run)
# ═══════════════════════════════════════════════════════════════

if [ $SKIP_SETUP -eq 0 ]; then

    log "Setting up Termux environment..."

    # ── 1. System packages ──
    log "Installing system packages..."
    pkg update -y 2>/dev/null || warn "pkg update had errors (usually harmless)"
    pkg install -y python git 2>/dev/null || warn "pkg install had errors"
    
    # numpy/scipy need these on Termux
    pkg install -y build-essential binutils 2>/dev/null || true
    pkg install -y python-numpy python-scipy 2>/dev/null || {
        warn "python-numpy/scipy not in pkg, will try pip"
    }
    ok "System packages"

    # ── 2. Python packages ──
    log "Installing Python packages..."
    pip install --upgrade pip 2>/dev/null || true
    
    # Try importing first (Termux pkg may have installed them)
    python3 -c "import numpy" 2>/dev/null || {
        log "Installing numpy via pip (this may take a few minutes)..."
        pip install numpy
    }
    python3 -c "import scipy" 2>/dev/null || {
        log "Installing scipy via pip (this may take 5-10 minutes)..."
        MATHLIB=m pip install scipy || {
            warn "scipy install failed. Trying with --no-build-isolation..."
            pip install scipy --no-build-isolation || {
                warn "scipy failed. Trying older version..."
                pip install "scipy<1.12" || fail "Cannot install scipy. Try: pkg install python-scipy"
            }
        }
    }
    
    # Verify
    python3 -c "import numpy; import scipy; print(f'numpy {numpy.__version__}, scipy {scipy.__version__}')" \
        && ok "Python packages" \
        || fail "Python package verification failed"

    # ── 3. Clone/update repo ──
    if [ -d "$WORK_DIR/.git" ]; then
        log "Updating repo..."
        cd "$WORK_DIR" && git pull --ff-only 2>/dev/null || true
        ok "Repo updated"
    else
        log "Cloning repo (this downloads ~15MB)..."
        git clone "$REPO_URL" "$WORK_DIR" 2>&1 | tail -3
        ok "Repo cloned"
    fi

    # ── 4. Verify data files ──
    log "Verifying data files..."
    [ -f "$WORK_DIR/enriched_records.pkl" ] || fail "enriched_records.pkl not found"
    [ -f "$PAPER_DIR/p70c_full_spec_v1.json" ] || fail "p70c_full_spec_v1.json not found"
    [ -f "$PAPER_DIR/score_85_metrics.py" ] || fail "score_85_metrics.py not found"
    [ -f "$PAPER_DIR/reproduce_s3.py" ] || fail "reproduce_s3.py not found"
    [ -f "$PAPER_DIR/reproduce_all.py" ] || fail "reproduce_all.py not found"
    [ -d "$PAPER_DIR/Generators" ] || fail "Generators/ directory not found"
    ok "All data files present"

    # ── 5. Quick sanity check ──
    log "Running sanity check..."
    cd "$PAPER_DIR"
    python3 -c "
import pickle, json, sys
with open('../enriched_records.pkl', 'rb') as f:
    r = pickle.load(f)
print(f'  Records: {len(r)} tokens')
with open('p70c_full_spec_v1.json') as f:
    s = json.load(f)
print(f'  P70C entries: {len(s[\"entries\"])}')
from score_85_metrics import TOLERANCES
print(f'  Tolerances: {len(TOLERANCES)} metrics')
print('  Sanity check PASSED')
" || fail "Sanity check failed"

    echo ""
    ok "Setup complete. Ready to run."
    echo ""

fi

# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

cd "$PAPER_DIR"

# Build the command
CMD="python3 reproduce_s3.py --resume"

if [ $QUICK -eq 1 ]; then
    CMD="$CMD --skip-bg22 --skip-ablation"
    log "Quick mode: skipping BG22 generators and ablation sweeps"
fi

CMD="$CMD $EXTRA_ARGS"

log "Running: $CMD"
log "Results will be saved to: $PAPER_DIR/results/s3/"
log "If interrupted, rerun with --run to resume from cache"
echo ""

# Set memory-friendly environment
export PYTHONDONTWRITEBYTECODE=1

# Acquire wakelock to prevent sleep during long runs
termux-wake-lock 2>/dev/null || true

START=$(date +%s)

# Run with nice to be friendly to other apps
nice -n 10 $CMD

EXIT=$?
END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

# Release wakelock
termux-wake-unlock 2>/dev/null || true

echo ""
if [ $EXIT -eq 0 ]; then
    ok "Completed in ${ELAPSED} minutes"
    echo ""
    log "Results in: $PAPER_DIR/results/s3/"
    ls -lh "$PAPER_DIR/results/s3/"*.pkl 2>/dev/null
    echo ""
    
    # Show summary if it exists
    if [ -f "$PAPER_DIR/results/s3/s3_summary.md" ]; then
        log "Summary:"
        head -50 "$PAPER_DIR/results/s3/s3_summary.md"
    fi
    
    # Send notification if Termux:API is installed
    termux-notification -t "S3 Complete" -c "Finished in ${ELAPSED}min" 2>/dev/null || true
else
    fail "Exited with code $EXIT after ${ELAPSED} minutes"
    warn "Rerun with: bash termux_reproduce_s3.sh --run"
    warn "(cached results are preserved)"
    termux-notification -t "S3 Failed" -c "Exit code $EXIT after ${ELAPSED}min" 2>/dev/null || true
fi
