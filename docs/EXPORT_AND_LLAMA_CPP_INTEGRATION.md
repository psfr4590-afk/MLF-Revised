# Export, llama.cpp, and Ollama Integration

This document maps the Phase 9 deployment chain: checkpoint → Hugging Face-compatible checkpoint → F16/Q4_K_M/Q5_K_M/Q8_0 GGUF → llama.cpp → Ollama.

## Architecture Overview

```
Training Pipeline
    ↓
checkpoint + provenance
    ↓
[export_gguf.py] ← validates provenance, loads checkpoint, trains HF → maps tensors → writes HF checkpoint
    ↓
[reconcile_environment.py] + [bootstrap_llama_cpp.sh]
    ↓ (clones & builds pinned llama.cpp)
third_party/llama.cpp/
    ├── convert_hf_to_gguf.py (Python converter)
    ├── build/llama-quantize (C++ quantizer)
    └── build/llama-cli (C++ inference CLI)
    ↓
[export_gguf.py continued]
    ├── convert_hf_to_gguf.py → model-f16.gguf
    ├── llama-quantize → model-Q4_K_M.gguf (or other quant)
    └── Modelfile (Ollama format)
    ↓
[verify_gguf.py]
    ├── finds llama-cli executable
    ├── runs: llama-cli -m model.gguf -p "Hello" -n 1
    └── captures output → updates export_manifest.json with inference_validation
    ↓
export_manifest.json + cards → release evidence
```

## Detailed Flow

### 1. Checkpoint Validation (`export_gguf.py:_enforce_training_provenance`)

Before export, the system validates:
- **Checkpoint provenance fields** (required): schema 3, run_id, independent configuration identities, train_config_sha256, model_config_sha256, shard_manifest_sha256, source_manifest_sha256, seed, and explicit upstream artifact IDs
- **Artifact chain integrity**:
  - shard_manifest.json matches shard_manifest_sha256
  - tokenizer.json.manifest.json matches tokenizer metadata
  - weighted corpus manifest matches pipeline config
  - source_manifest.json matches source_manifest_sha256
- **Tokenizer/model vocab match**: tokenizer vocabulary size == model.vocab_size

If any validation fails, export refuses to proceed. This is **fail-closed**.

### 2. Checkpoint Loading and Tensor Mapping (`export_gguf.py:_load_checkpoint`, `_map_state`)

The Model Lab checkpoint uses an internal tensor naming scheme that must be mapped to Hugging Face naming:
- `embed.weight` → `model.embed_tokens.weight`
- `norm.weight` → `model.norm.weight`
- `lm_head.weight` → `lm_head.weight` (unchanged)
- `blocks.[i].norm1.weight` → `model.layers.[i].input_layernorm.weight`
- `blocks.[i].norm2.weight` → `model.layers.[i].post_attention_layernorm.weight`
- `blocks.[i].attn.q_proj.weight` → `model.layers.[i].self_attn.q_proj.weight`
- (similarly for k_proj, v_proj, out_proj)
- (similarly for ffn gate, up, down projections)

Rope embeddings (`rope_cos`, `rope_sin`) are excluded; they are regenerated at inference time.

**Critical check**: embedding and lm_head shapes must match (tied embeddings).

### 3. Hugging Face Checkpoint Write (`export_gguf.py:_write_hf_checkpoint`)

Writes to `output/export/hf/`:
- `pytorch_model.bin` — mapped state dict
- `config.json` — Hugging Face config (architecture=LlamaForCausalLM, vocab_size, etc.)
- `tokenizer.json` — (copied from output/tokenizer/)
- `tokenizer_config.json` — (copied if present)
- `special_tokens_map.json` — (copied if present)

This directory is a valid Hugging Face llama checkpoint and can be loaded with transformers.

### 4. llama.cpp Toolchain Bootstrap (`scripts/reconcile_environment.py`, `scripts/bootstrap_llama_cpp.sh`)

**Trigger**: `scripts/verify_release.py --bootstrap-native` calls `reconcile_environment.py --ensure-llamacpp`

**What it does**:
1. Clones llama.cpp at pinned tag `b10516` (commit `b95502b`)
2. Verifies exact commit hash (fail-closed if mismatch)
3. Checks for `convert_hf_to_gguf.py` (fail-closed if missing)
4. Builds with CMake:
   - Release mode, static linking, no CUDA/OpenMP/curl
   - For Android/Termux: single-threaded, `llama-quantize` only
   - For desktop: parallel build, both `llama-quantize` and `llama-cli`
5. Applies tokenizer compatibility patch (`patch_llama_cpp_tokenizer.py`)

**Tokenizer Compatibility Patch**:
- Model Lab uses `tokenizers.ByteLevel(add_prefix_space=False)` for BPE
- llama.cpp's converter historically matches pre-tokenizers by hash
- A newly trained vocab produces a new hash even with identical behavior
- The patch adds structural detection: if llama.cpp sees ByteLevel, use GPT-2-compatible pre-tokenizer
- **Fails closed**: if the expected `get_vocab_base_pre` marker is absent, the patch fails

### 5. HF → GGUF Conversion (`export_gguf.py:_convert_to_f16`, `_quantize`)

**Step 1: F16 Conversion**
```bash
python convert_hf_to_gguf.py <hf_checkpoint_dir> --outfile model-f16.gguf --outtype f16
```
- Uses the patched llama.cpp converter
- Outputs 16-bit floating point GGUF
- Requires valid Hugging Face checkpoint structure

**Step 2: Quantization (optional)**
```bash
llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```
- Supported quantizations: F16, Q4_K_M, Q5_K_M, Q8_0
- Default: Q4_K_M (4-bit K-means, medium quality/speed tradeoff)
- Outputs smaller, faster-running artifact

**Critical checks**:
- Output file exists and is non-empty (fail-closed)
- Executable exists before running (fail-closed)

### 6. Export Manifest and Cards (`export_gguf.py:export_checkpoint`, `scripts/export_cards.py`)

Writes `output/gguf/export_manifest.json`:
```json
{
  "schema": 3,
  "checkpoint": "path/to/ckpt_best_1000.pt",
  "checkpoint_step": 1000,
  "model_name": "pretrain-model",
  "model_config": { ... },
  "quantization": "Q4_K_M",
  "llamacpp_dir": "path/to/third_party/llama.cpp",
  "llamacpp_version": "b10516@b95502b",
  "final_gguf": "path/to/model-Q4_K_M.gguf",
  "final_gguf_sha256": "<sha256 hash>",
  "dataset_card": "path/to/DATASET_CARD.md",
  "dataset_card_sha256": "<sha256 hash>",
  "model_card": "path/to/MODEL_CARD.md",
  "model_card_sha256": "<sha256 hash>",
  "inference_validation": { ... }  // populated by verify_gguf.py
}
```

**Export Cards** (`DATASET_CARD.md`, `MODEL_CARD.md`):
- Auditable provenance: training config, dataset stats, shard counts, tokenizer hash
- Source manifest reference
- Git commit of the training run
- Quantization method and model size

### 7. GGUF Inference Validation (`scripts/verify_gguf.py`)

Phase 9 validates GGUF metadata before generation, including architecture, embedded context length, tokenizer metadata, and BOS/EOS/PAD special-token IDs when expected values are supplied. It also requests llama.cpp tensor validation before generation.

**Input**: Final GGUF artifact and llama-cli executable

**Execution**:
```bash
llama-cli -m model.gguf -p "Hello" -n 1 --single-turn --no-display-prompt --simple-io
```

**Checks**:
- GGUF file exists and is non-empty
- llama-cli executable exists
- Command executes without error (timeout: 120s)
- At least one token was generated (stdout is non-empty)

**Output**: Stores in export_manifest.json:
```json
{
  "inference_validation": {
    "status": "passed",
    "model": "path/to/model.gguf",
    "llama_cli": "path/to/llama-cli",
    "prompt": "Hello",
    "generated_output": "<model output>"
  }
}
```

**Fail condition**: If status != "passed" OR no generated_output, release gate fails.

## Release Gate Sequence (`scripts/verify_release.py --bootstrap-native`)

1. **Static checks** (compile, lint, tests, doctor)
2. **Create deterministic smoke fixture**:
   - 64 documents across 2 source families
   - Local-only metadata, no network calls
   - 04_weighted.jsonl + manifests
3. **Build llama.cpp** via `reconcile_environment.py --ensure-llamacpp`
   - Clone, verify commit, build both quantizer and CLI
   - Apply tokenizer patch
4. **Run smoke pipeline**:
   - Load weighted fixture
   - Run: crawl=OFF, clean=OFF, dedup=OFF, weight=OFF, tokenize=ON, shard=ON, train=ON, export=ON
   - Outputs checkpoint, tokenizer, shards
5. **Export to GGUF**:
   - Validate provenance chain
   - Write HF checkpoint
   - Convert to F16
   - Quantize to Q4_K_M
   - Generate export cards
6. **Verify GGUF with llama-cli**:
   - Load model
   - Generate one token
   - Capture output
7. **Check inference_validation in manifest**:
   - status must be "passed"
   - generated_output must be non-empty
8. **Success**: Return 0 (release candidate validated)

## Key Assumptions and Guarantees

| Aspect | Guarantee |
|--------|-----------|
| **llama.cpp version** | Pinned to exact commit (b95502b), reproducible across platforms |
| **Tokenizer compat** | Patch applied at reconciliation time, fails closed if llama.cpp internals change |
| **Provenance chain** | Artifacts are hashed with SHA-256; export checks recorded hashes and lineage links. Hashes detect changes but are not digital signatures. |
| **Tensor mapping** | Deterministic Llama → HF → GGUF, validated against schema |
| **Quantization** | Reproducibility depends on pinned converter/toolchain versions and execution environment; verify output hashes rather than assuming bit-for-bit identity across hosts. |
| **Inference test** | Single-token generation; proves model loads and can generate output |
| **Export cards** | Auditable markdown; not used for verification, but retained as evidence |

## Ollama Deployment Validation

`scripts/verify_ollama.py` validates the exported GGUF through an external Ollama runtime. It creates a temporary model from the generated Modelfile, loads it, runs generation, and verifies `ollama show` succeeds. The full native release gate can run this for every quantization with `--verify-ollama`.

## Known Limitations

1. **Inference test is an execution gate**: It proves loading, tokenizer metadata, tensor integrity, context configuration, and generation. It does not prove quality or convergence
2. **Quantization is lossy**: Q4_K_M reduces precision; users should test on their target tasks
3. **Native gate is slow**: llama.cpp build can take 5–30 minutes depending on platform
4. **Android/Termux**: Serialized build (single-threaded) may take even longer
5. **Ollama is host-dependent**: Ollama verification requires a separately installed Ollama runtime and is intentionally an explicit deployment-gate option.
6. **Release evidence is platform-specific**: The native gate output includes platform, Python version, and build-time facts; final release may require re-runs on target platforms

## Files and Entry Points

| File | Role |
|------|------|
| `scripts/export_gguf.py` | Main export logic (checkpoint → HF → GGUF) |
| `scripts/verify_gguf.py` | Inference validation (GGUF + llama-cli → output) |
| `scripts/verify_release.py` | Master release gate; orchestrates static + native checks |
| `scripts/reconcile_environment.py` | Bootstrap native toolchain |
| `scripts/bootstrap_llama_cpp.sh` | Clone and build llama.cpp (bash) |
| `scripts/bootstrap_llama_cpp.ps1` | Clone and build llama.cpp (PowerShell) |
| `scripts/patch_llama_cpp_tokenizer.py` | Apply tokenizer compatibility patch |
| `scripts/export_cards.py` | Generate auditable dataset/model cards |
| `third_party/llama.cpp/` | Cloned at reconciliation; not in repo |

## Troubleshooting

### "llama.cpp converter not found"
- Run `scripts/verify_release.py --bootstrap-native` first to clone and build
- Or manually clone with the pinned tag and commit

### "llama-quantize was not built"
- Check CMake build output; may be missing C++ compiler or CMake
- On Windows, ensure Visual Studio is installed
- On Linux/macOS, ensure gcc/clang and cmake are in PATH

### "llama-cli failed with exit code X"
- GGUF may be corrupted; re-export from checkpoint
- Or llama-cli may have crashed on the model; check stderr in verify_gguf.py
- Verify GGUF file size is reasonable (typically 2–15 GB depending on quantization)

### "Tokenizer vocab does not match model vocab_size"
- Checkpoint was trained with one tokenizer; GGUF was exported with a different one
- Verify the tokenizer.json in the output directory matches the checkpoint

### "Checkpoint provenance is incomplete"
- Missing required fields in checkpoint payload
- Re-run training with a current version; older checkpoints may not have full provenance

## Next Steps

For release validation, follow:
```bash
python scripts/verify_release.py --bootstrap-native
```

This command will:
1. Run all static checks
2. Clone and build the pinned llama.cpp toolchain
3. Run the deterministic smoke pipeline
4. Export to GGUF and verify inference
5. Report success or failure with platform details
