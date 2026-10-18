# Dataset Card

## Dataset summary

- Dataset name: `[DATASET_NAME]`
- Dataset version/run: `[DATASET_RUN_ID]`
- Description: `[DESCRIPTION]`
- Intended use: `[INTENDED_USE]`
- Out-of-scope use: `[OUT_OF_SCOPE_USE]`

## Composition

- Documents after crawl: `[CRAWL_DOCUMENTS]`
- Documents after cleaning: `[CLEAN_DOCUMENTS]`
- Documents after deduplication: `[DEDUP_DOCUMENTS]`
- Documents after weighting: `[WEIGHTED_DOCUMENTS]`
- Tokenizer vocabulary: `[VOCAB_SIZE]`
- Sharded tokens: `[SHARDED_TOKENS]`

## Sources and provenance

Every source used for a production run must have an auditable source manifest. Public availability does not by itself establish permission to use content for training.

- Source manifest: `[SOURCE_MANIFEST]`
- Source manifest SHA-256: `[SOURCE_MANIFEST_SHA256]`
- Pipeline configuration SHA-256 (legacy compatibility evidence): `[PIPELINE_CONFIG_SHA256]`
- Dataset configuration SHA-256: `[DATASET_CONFIG_SHA256]`
- Tokenizer configuration SHA-256: `[TOKENIZER_CONFIG_SHA256]`
- Shard configuration SHA-256: `[SHARD_CONFIG_SHA256]`
- Source-definition configuration SHA-256: `[SOURCE_DEFINITION_SHA256]`
- Git commit: `[GIT_COMMIT]`

For each source, retain its URL or identifier, retrieval time, revision, license or usage terms, attribution requirements, operator restrictions, and raw-source SHA-256.

## Processing

`crawl → clean → semantic dedup → weight → tokenize → shard`

Record any filtering, normalization, deduplication, weighting, rejection, or exclusion policy that materially changes the corpus.

## Licensing and rights

`[LICENSES_AND_USAGE_TERMS]`

The framework records technical provenance; it does not certify copyright ownership, consent, licensing, or legal compliance.

## Quality and limitations

`[QUALITY_SUMMARY]`

Document known gaps, language/domain skew, duplication risk, noisy sources, extraction failures, and any evaluation limitations.

## Maintenance and removal

If a source must be removed, disable it in the source manifest and rebuild affected downstream artifacts. Do not edit generated training data in place without regenerating its provenance chain.
