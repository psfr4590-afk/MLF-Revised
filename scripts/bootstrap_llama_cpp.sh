#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/third_party/llama.cpp"
REPO="https://github.com/ggml-org/llama.cpp.git"
TAG="b10516"
COMMIT="b95502b"
CUDA="OFF"
for arg in "$@"; do
  case "$arg" in
    --cuda) CUDA="ON" ;;
    --force) echo "--force is not supported because bootstrap must preserve existing checkout data." >&2; exit 2 ;;
    *) echo "Unknown argument: $arg (supported: --cuda)" >&2; exit 2 ;;
  esac
done

if [[ -d "$TARGET/.git" ]]; then
  echo "llama.cpp checkout already exists. Verifying existing checkout..."
else
  if [[ -e "$TARGET" ]]; then
    echo "Refusing to delete or overwrite an existing directory without a Git checkout: $TARGET. Inspect it and move or back it up manually before retrying." >&2
    exit 2
  fi
  mkdir -p "$(dirname "$TARGET")"
  git clone --depth 1 --branch "$TAG" "$REPO" "$TARGET"
fi

cd "$TARGET"
actual="$(git rev-parse HEAD)"
[[ "$actual" == "$COMMIT"* ]] || { echo "llama.cpp pin mismatch: expected $COMMIT got $actual" >&2; exit 1; }
test -f convert_hf_to_gguf.py || { echo "Missing convert_hf_to_gguf.py" >&2; exit 1; }

BUILD_ARGS=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON -DGGML_CUDA="$CUDA")

# Android/Termux uses a smaller native profile.
if [[ "$(uname -o 2>/dev/null || true)" == "Android" || -n "${TERMUX_VERSION:-}" ]]; then
  if [[ "$CUDA" == "ON" ]]; then
    echo "CUDA build is not supported by the minimal Android/Termux profile." >&2
    exit 2
  fi
  echo "Detected Termux/Android; using minimal llama.cpp build profile."
  BUILD_DIR="build-model-lab"
  cmake -S . -B "$BUILD_DIR" "${BUILD_ARGS[@]}"
  # Keep the Android build serialized to reduce memory pressure during compilation.
  cmake --build "$BUILD_DIR" --config Release --parallel 1 --target llama-quantize
  quantizer="$(find "$BUILD_DIR" -type f -name 'llama-quantize*' -print -quit)"
  test -n "$quantizer" || { echo "llama-quantize was not built" >&2; exit 1; }
  echo "llama.cpp ready at $TARGET"
  echo "  converter: $TARGET/convert_hf_to_gguf.py"
  echo "  quantizer: $quantizer"
else
  BUILD_DIR="build"
  cmake -S . -B "$BUILD_DIR" "${BUILD_ARGS[@]}"
  cmake --build "$BUILD_DIR" --config Release --parallel --target llama-quantize llama-cli llama-app
  quantizer="$(find "$BUILD_DIR" -type f -name 'llama-quantize*' -print -quit)"
  cli="$(find "$BUILD_DIR" -type f -name 'llama-cli*' -print -quit)"
  app="$(find "$BUILD_DIR" -type f \( -name 'llama.exe' -o -name 'llama' \) -print -quit)"
  test -n "$quantizer" || { echo "llama-quantize was not built" >&2; exit 1; }
  test -n "$cli" || { echo "llama-cli was not built" >&2; exit 1; }
  test -n "$app" || { echo "llama-app (llama.exe/llama) was not built" >&2; exit 1; }
  "$app" --help >/dev/null || { echo "llama-app --help failed" >&2; exit 1; }
  "$cli" --help >/dev/null || { echo "llama-cli --help failed" >&2; exit 1; }
  "$quantizer" --help >/dev/null || { echo "llama-quantize --help failed" >&2; exit 1; }
  echo "llama.cpp ready at $TARGET (GGML_CUDA=$CUDA)"
  echo "  converter: $TARGET/convert_hf_to_gguf.py"
  echo "  quantizer: $quantizer"
  echo "  cli: $cli"
  echo "  app: $app"
fi
