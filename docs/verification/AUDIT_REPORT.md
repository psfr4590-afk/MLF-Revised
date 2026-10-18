# Model Lab Release Gate Checklist

> **Policy/checklist, not an audit result.** This document defines the evidence required before calling a build production-ready. It does not claim that every gate listed below has been executed for the current repository state.

Release approval requires automated tests, the non-destructive environment doctor, a native llama.cpp bootstrap, and a reduced end-to-end training/export smoke test on the target machine. The project must not claim operations that were not actually executed.

## Mandatory release gates

- `compileall`
- `pytest`
- non-destructive environment doctor
- native llama.cpp bootstrap
- reduced corpus run
- checkpoint reload
- GGUF conversion
- quantization when requested
- manifest hash verification
- llama.cpp inference smoke test

Each gate should be recorded with its execution environment, result, and relevant artifact or log evidence before release approval. Current automated CI status is separate from target-machine native verification.
