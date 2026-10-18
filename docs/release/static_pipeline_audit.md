# Static Pipeline Release Audit

This audit defines what can be established from the repository without performing a production crawl or long training run.

## 1. Clone and repository identity

- The project root is resolved from the entry-point file rather than the caller's working directory.
- Python support is explicitly constrained to 3.11 through 3.14 by `pyproject.toml`, `bootstrap.py`, and the runtime doctor.
- The bootstrapper has a separate PyTorch installation path so CUDA and CPU wheels are selected intentionally.
- Native llama.cpp is not assumed to exist in a clean clone. The reconciler installs a pinned checkout and verifies the expected revision before building it.
- Generated datasets, checkpoints, GGUF files, and other runtime artifacts are not part of the repository contract.

## 2. Dependency and environment gate

- `bootstrap.py --doctor` checks Python, core imports, Git, CMake, hardware, and actual Torch CUDA availability.
- Runtime dependency authority is `pyproject.toml`; `requirements.txt` is the editable-install adapter and host-specific requirement files remain separate.
- The release security gate scans tracked files for common credential patterns and can run `pip-audit` against the declared dependency sets.

## 3. Configuration identity

Every checked-in pipeline configuration is validated against the strict configuration schema. The dataset-session configuration explicitly declares the schema-2 canonical dataset profile catalog.

The canonical dataset catalog is:

| ID | Group |
|---:|---|
| 1 | `swe_cs_systems` |
| 2 | `ai_ml_cybersec_dataeng` |
| 3 | `sci_reasoning_forensics_formal` |
| 4 | `domain_finance_bio_robotics` |
| 5 | `math_statistics_optimization` |
| 6 | `physics_chemistry_materials` |
| 7 | `biomedical_health_science` |
| 8 | `law_compliance_governance` |
| 9 | `linguistics_information_retrieval` |
| 10 | `climate_energy_geospatial` |

Canonical IDs are contiguous, unique, and bound one-to-one to group IDs. Command Center creation and dataset seeding refuse identity collisions.

## 4. Source provenance

Source manifests use schema 3 and record (schema 2 remains readable by the independent lineage validator):

- selected dataset group;
- retrieval start and completion timestamps;
- source-definition file hashes;
- source kind and identifier;
- revision and license fields;
- raw-source SHA-256 when available;
- dataset-group identity for every source row.

An incomplete or stale source manifest cannot satisfy the production export contract.

## 5. Stage lineage

The static contract is:

`crawl -> clean -> semantic dedup -> weight -> tokenize -> shard -> train -> export`

Each material JSONL artifact receives an integrity manifest containing size, SHA-256, kind, row count, and stage provenance. Resume logic skips an artifact only when the bytes and expected provenance still match.

The shard manifest independently hashes every shard and records the stage provenance. Tokenizer and weighted-corpus manifests are checked before export.

## 6. Training lineage

Training checkpoints contain:

- model configuration;
- training configuration;
- optimizer/scaler state;
- deterministic RNG state;
- deterministic train/validation loader state;
- checkpoint integrity metadata;
- pipeline configuration identity;
- model configuration identity;
- training configuration identity;
- shard manifest identity;
- exact source-manifest identity;
- seed.

Production dataset sessions refuse accidental CPU pretraining. Hardware-aware auto-sizing remains available, but it cannot silently turn a CUDA-required production session into a CPU run.

## 7. Export lineage

Before conversion, export verifies the checkpoint itself and then verifies the retained source manifest, weighted corpus manifest, tokenizer manifest, and shard manifest. The checkpoint's recorded source-manifest SHA-256 must equal the current source manifest.

The export path then:

1. maps the checkpoint into a local Hugging Face-style model directory;
2. copies the tokenizer artifacts;
3. converts the model to F16 GGUF with the pinned llama.cpp converter;
4. quantizes to the requested format when applicable;
5. writes an Ollama `Modelfile`;
6. writes dataset and model cards;
7. records final artifact sizes and SHA-256 values;
8. records training provenance and converter revision.

## 8. Native release verification

`scripts/verify_release.py --bootstrap-native` is the final static-to-native gate. It compiles the project, runs Ruff, runs the test suite with coverage, runs the Doctor checks, reconciles the pinned llama.cpp toolchain, executes a bounded network-free fixture through tokenization, sharding, training, GGUF export, export-card generation, and native llama.cpp inference.

A successful static audit does not claim that a production dataset has been crawled or that a production model has been trained. Those are runtime facts and must be established by the actual run.

## 9. Remaining runtime-only evidence

The repository can prove the contracts above, but it cannot statically prove:

- current availability or content of remote sources;
- HTTP responses, robots rules, or API quotas at run time;
- the quality or legal status of every retrieved source beyond recorded metadata;
- actual training loss, convergence, or model capability;
- successful native conversion on the current host until the native release gate is executed.

Those are deliberately left as runtime gates instead of being simulated with fake artifacts. Humanity has suffered enough from green checkmarks generated by optimism.
