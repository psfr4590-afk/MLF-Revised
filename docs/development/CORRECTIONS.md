# Corrections applied to the reference ZIP

The original `ML-Framework-main.zip` extraction is preserved unmodified as the read-only
reference. Everything below was changed in this working copy only. Status labels follow the
project vocabulary: **verified** (an executed test exercised it and passed), **implemented**
(code exists, not yet exercised end to end), **blocked** (cannot be verified in the current
environment).

## Defects found and fixed

| # | Defect (evidence) | Fix | Status |
|---|---|---|---|
| 1 | **Exporter exported the wrong checkpoint.** Trainer writes `ckpt_final.pt`, `ckpt_best.pt`, `ckpt_0000010.pt`; exporter globbed `ckpt_best_*.pt` / `ckpt_final_*.pt` (never match), fell through to `ckpt_*.pt` and exported the highest-numbered *periodic* checkpoint. Its tests used the same wrong names, so everything "passed". | `pipeline.integrity.select_export_checkpoint`: final → best → latest, each manifest-verified; legacy names last. Orchestrator now passes the checkpoint its train stage selected. | verified (torch-free tests); orchestrator pass-through implemented |
| 2 | **Command Center never showed training complete.** Globbed `ckpt_final_*.pt` (nonexistent) and validated with `artifact_valid`, which requires manifest `schema == 2` while checkpoint manifests are schema 3. | `checkpoint_manifest_valid` (schema ≥ 2, kind, size, SHA-256) and correct glob. | verified |
| 3 | **Hardcoded special-token IDs** (`bos=2, eos=3, pad=0`) in the HF config, release gate and docs; wrong for any other `special_tokens` order. | `pipeline/tokenizer/special_tokens.py` resolves IDs from the real tokenizer; used by exporter, `verify_release.py`, `mlframework infer`. IDs recorded in the export manifest. | verified (resolver); wiring implemented |
| 4 | **No non-finite loss guard** (NaN would be checkpointed, written as invalid JSON, then exported). | `pipeline/numerics.require_finite` on every micro-batch loss and every validation loss. | guard verified; trainer wiring implemented |
| 5 | **GGUF verifier did not validate tensor data.** No tensor-table parse, no truncation detection, unbounded reads on corrupt headers, no required-tensor/shape checks, no determinism check. | Bounds-checked reader, tensor table + data-completeness, llama tensor-set/shape/vocab checks, `--temp 0 --seed`, repeat-run comparison (CLI default 2), evidence files. | verified |
| 6 | `mlframework infer` skipped the context-length and special-token checks the release gate enforces. | Reads both from the export manifest. | implemented |

## Investigated and found sound (no change)

* **RoPE convention.** The model uses split-half rotation; this is consistent because the
  exporter writes a Hugging Face `LlamaForCausalLM` checkpoint and llama.cpp's own converter
  applies the Q/K permutation.
* **Context-size propagation.** `tokenizer.vocab_size == train.vocab_size` and
  `shard.sequence_length == train.seq_len` are enforced in `config_validation.py`.
* **`bootstrap.py` torch-channel logic** (`_remove_incompatible_torch`) is validated after install.

## Not verified here (blocked in the authoring sandbox)

The sandbox had no PyTorch (disk), no llama.cpp build and no tkinter. Therefore **not run**:
the trainer, real checkpoint save/load/resume, `export_gguf` against llama.cpp, real GGUF
inference, A1–A7 acceptance profiles, clean-install on Windows. 11 test modules need torch /
tkinter and 5 more tests need sentence-transformers; they failed identically before and after
these changes. Run on the target machine:

```powershell
python -m pytest
python scripts\verify_release.py --bootstrap-native
```

Nothing in this working copy is **release-certified**.
