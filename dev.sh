#!/usr/bin/env bash
# Usage: ./dev.sh [shader.glsl] [extra shadertoy flags]
# Editor on the left, live render on the right. Creates the file if missing.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Resolve the shader path against the caller's cwd BEFORE changing directory.
SHADER="${1:-$ROOT/examples/plasma.glsl}"; shift || true
case "$SHADER" in /*) ;; *) SHADER="$PWD/$SHADER" ;; esac
if [ ! -e "$SHADER" ]; then
  mkdir -p "$(dirname "$SHADER")"
  cat > "$SHADER" <<'GLSL'
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    fragColor = vec4(0.5 + 0.5 * cos(iTime + uv.xyx + vec3(0, 2, 4)), 1.0);
}
GLSL
  echo "created $SHADER"
fi

(cd "$ROOT" && cargo build --release -q)
BIN="$ROOT/target/release/shadertoy"
# On failure keep the pane open so the error is readable.
RUN="$BIN '$SHADER' $*; echo; echo '[renderer exited - press enter]'; read"
ED="${EDITOR:-hx}"

if [ -n "$TMUX" ]; then
  tmux split-window -h "$RUN"
  tmux select-pane -L
  exec $ED "$SHADER"
else
  tmux kill-session -t shadertoy 2>/dev/null || true
  tmux new-session -d -s shadertoy "$ED '$SHADER'"
  tmux split-window -h -t shadertoy "$RUN"
  tmux select-pane -t shadertoy.0
  exec tmux attach -t shadertoy
fi
