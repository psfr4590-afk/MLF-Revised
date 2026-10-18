# Dataset provenance and source policy

Model Lab can technically collect and transform data, but the software cannot grant permission to use someone else's content for training. Operators are responsible for confirming that every source used by a run is lawful and appropriate for the intended dataset and model.

## What the pipeline records

Every stage writes an integrity manifest. New Phase 1 manifests carry a run identity, independent dataset/tokenizer/shard/training/model configuration identities, implementation identity, input hashes, and explicit parent artifact relationships. The historical pipeline configuration hash is retained as legacy evidence, not as a universal identity. This prevents an artifact produced from one corpus/configuration from being silently reused as if it came from another.

For a production dataset, retain a source manifest containing at least:

- source URL or dataset/repository identifier
- retrieval timestamp
- source type
- source revision/version when available
- applicable license or usage terms
- attribution requirements
- any operator approval or restriction notes
- raw-source artifact SHA-256
- the Git commit, source-definition configuration identity, and exact configuration snapshot used to process it

## Export cards

Every successful GGUF export now emits two auditable Markdown artifacts beside `export_manifest.json`:

- `DATASET_CARD.md` — records the dataset processing lineage, available stage counts, tokenizer/shard hashes, independent configuration identities, and source-definition fingerprint, Git revision, and the boundary between technical provenance and legal rights.
- `MODEL_CARD.md` — records model/export configuration, checkpoint and training provenance, seed where available, GGUF hashes, llama.cpp revision, and explicit evaluation/safety limitations.

The generated cards are referenced and SHA-256 protected by `export_manifest.json`. Repository-level starting templates are maintained at `docs/templates/DATASET_CARD.md` and `docs/templates/MODEL_CARD.md`.

Unknown provenance or evaluation values are represented as unknown rather than fabricated. A card is evidence packaging, not a substitute for an operator's source-rights review or model evaluation.

## Licensing and terms

The seeded configuration is an engineering starting point, not a blanket license grant. Before a large or redistributed training run, review the terms for each configured web site, GitHub repository, arXiv source, Hugging Face dataset, search result, or other provider.

Do not assume that a public URL means unrestricted training rights. In particular, review dataset licenses, repository licenses, website terms, robots/publishing policies, attribution requirements, and any restrictions on redistribution or commercial use.

If the required rights or provenance cannot be established, exclude the source from the production run rather than relying on the crawler to make a legal determination.

## Removal and takedown

A production dataset should be rebuildable from source manifests. When a source must be removed, disable it in the source configuration, record the reason in the dataset provenance record, rebuild affected downstream artifacts, and preserve the resulting artifact hashes. Do not patch a generated training corpus in place without regenerating its provenance chain.

## Data quality

Provenance is separate from quality. A legally usable source can still be poor training data. Production runs should retain counts for acquisition, cleaning, deduplication, weighting, tokenization, and sharding, along with rejection reasons where available. Large changes in these counts should be reviewed before training.

## Boundary of responsibility

Model Lab provides technical controls for bounded acquisition, parsing, filtering, deduplication, weighting, and artifact integrity. It does not certify copyright ownership, licensing, consent, or legal compliance for a dataset. Those decisions remain with the operator or organization running the pipeline.
