# Dataset Acceptance Contract

This contract is the gate between source configuration and training data. It does not crawl remote sources by itself.

## Static contract

`scripts/verify_dataset_contract.py` validates all ten canonical dataset profiles and groups before collection. It checks:

- exactly ten dataset IDs, 1 through 10
- one-to-one profile-to-group identity
- executable rules for every declared exclusion
- source-specific acquisition configuration
- explicit source identity fields
- bounded source counts and non-empty source definitions
- explicit Hugging Face revisions

Mutable Hugging Face revisions such as `main` are reported as non-reproducible. They are allowed for collection, but they are not considered production-revision-ready for a reproducible training release.

## Source identity

Every accepted corpus record must retain a source identity appropriate to its source kind:

- web: canonical URL
- GitHub: repository, default branch, and commit SHA when available
- arXiv: arXiv identifier and version
- Hugging Face: repository, revision, config, split, and row index
- Google search: result URL and query

The record's canonical JSON representation is also hashed by the crawler provenance layer.

## Rights and licensing

Rights metadata is retained on source records. Unknown rights do not automatically prevent collection, because discovery and legal acceptance are separate operations. Unknown rights do prevent a corpus from being considered distribution-ready. `--require-rights` makes the post-crawl acceptance gate fail on any unresolved rights record.

## Exclusions and quality

Profile exclusions are backed by executable regex rules in `config/dataset_source_policy.yaml`. They are not merely descriptive YAML labels anymore.

`scripts/verify_dataset_corpus.py` rejects:

- duplicate document IDs
- wrong dataset-group identity
- unsupported source kinds
- missing source identity
- documents below the minimum text-quality threshold
- documents matching an active profile exclusion

The quality gate is deliberately conservative and records rejection reasons rather than silently dropping evidence.

## Required workflow

1. Run `scripts/verify_dataset_contract.py` without crawling.
2. Resolve any static contract failure.
3. Crawl one dataset group into its isolated session.
4. Run `scripts/verify_dataset_corpus.py` against the crawl output.
5. Review rejection counts and unresolved rights.
6. Only then allow clean, semantic deduplication, weighting, tokenization, sharding, and training.
7. For a reproducible release, require immutable source revisions and verified rights.

This gate proves configuration and corpus integrity properties. It does not prove scientific quality, legal clearance of every source, or model capability.
