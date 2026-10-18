# Model evaluation

Phase 8 separates training completion from model quality.

Run:

    python scripts/evaluate_model.py --checkpoint best

The evaluator verifies the checkpoint integrity manifest, evaluates the held-out val shards, computes validation loss and perplexity, runs a fixed prompt suite across general, technical, science, history, and code domains, and records repetition and diversity metrics.

It writes:

- output/evaluations/model_evaluation.json
- output/evaluations/MODEL_EVALUATION.md

The report also compares available best, final, and latest checkpoints using their persisted validation evidence.

Generation labels are deliberately conservative. PASS means the generated text is non-empty and language-like under the mechanical checks. OBSERVED means the simple coherence heuristics were not triggered. These are not human-quality or safety certifications.

Every release candidate should retain the JSON report and Markdown report with the checkpoint SHA-256. Training success is not model-quality evidence.
