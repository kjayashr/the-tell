#!/usr/bin/env bash
# Installs Manim (community edition) + its system deps, into the-tell's venv.
# Safe to re-run. Run:  bash ~/Desktop/the-tell/install_manim.sh
set -uo pipefail

BLUE=$'\033[1;34m'; GRN=$'\033[1;32m'; RED=$'\033[1;31m'; NC=$'\033[0m'
say(){ echo "${BLUE}==>${NC} $*"; }
ok(){  echo "${GRN}  ok:${NC} $*"; }
die(){ echo "${RED}  error:${NC} $*"; exit 1; }

PROJ="$HOME/Desktop/the-tell"
VENV="$PROJ/.venv"

# 1) Homebrew
say "Checking Homebrew"
command -v brew >/dev/null 2>&1 || die "Homebrew not found. Install from https://brew.sh then re-run."
ok "brew at $(command -v brew)"

# 2) System libraries Manim needs (video encode + text/vector rendering)
say "Installing system deps (ffmpeg, cairo, pango, pkg-config) , idempotent, may take a few min"
brew install ffmpeg cairo pango pkg-config || die "brew install failed"
ok "system deps installed"

# 3) Python venv
say "Locating Python venv"
if [ ! -x "$VENV/bin/python" ]; then
  say "No venv at $VENV , creating one on python3.13"
  /opt/homebrew/bin/python3.13 -m venv "$VENV" || die "could not create venv"
fi
PY="$VENV/bin/python"
ok "using $PY ($($PY --version 2>&1))"

# 4) Manim (community edition) + helpers
say "Installing manim into the venv (this pulls pycairo/manimpango, compiled against the brew libs)"
export PKG_CONFIG_PATH="$(brew --prefix)/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q manim || die "pip install manim failed (see output above)"
ok "manim installed"

# 5) Verify + smoke-test render (static PNG, no LaTeX needed)
say "Verifying"
"$PY" -m manim --version || die "manim did not import"

say "Smoke-test render (a static frame to PNG)"
SCENE="$PROJ/_manim_smoketest.py"
cat > "$SCENE" <<'PYEOF'
from manim import *
class Smoke(Scene):
    def construct(self):
        self.add(Circle().set_stroke(BLUE, 6), Text("manim works").next_to(ORIGIN, DOWN))
PYEOF
cd "$PROJ" || die "cannot cd $PROJ"
# -s = save last frame as image; -o names it; --format png
"$PY" -m manim -s -o smoketest --format=png "$SCENE" Smoke \
  && ok "render succeeded" || die "render failed , check the manim output above"

OUT=$(find "$PROJ/media" -name "smoketest*.png" 2>/dev/null | head -1)
rm -f "$SCENE"
echo
echo "${GRN}=========================================${NC}"
echo "${GRN} Manim is installed and rendering.${NC}"
echo "${GRN}=========================================${NC}"
[ -n "$OUT" ] && echo " test image: $OUT"
echo
echo " To render a static diagram frame for the paper:"
echo "   $PY -m manim -s --format=png yourscene.py YourScene"
echo
echo " Note: LaTeX math mobjects (MathTex) need a TeX distro with dvisvgm."
echo " If you want those later:  brew install --cask basictex  &&  sudo tlmgr install dvisvgm"
echo " (Plain Text() mobjects work without it.)"
