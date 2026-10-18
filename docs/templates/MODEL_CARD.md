# Model Card

## Model details

- Model name: `[MODEL_NAME]`
- Export format: `[EXPORT_FORMAT]`
- Quantization: `[QUANTIZATION]`
- Checkpoint step: `[CHECKPOINT_STEP]`
- Model configuration: `[MODEL_CONFIG]`

## Intended use

`[INTENDED_USE]`

## Out-of-scope use

`[OUT_OF_SCOPE_USE]`

## Training data

- Dataset card: `[DATASET_CARD]`
- Dataset/provenance manifest: `[DATASET_PROVENANCE]`
- Training provenance: `[TRAINING_PROVENANCE]`
- Pipeline configuration SHA-256 (legacy compatibility evidence): `[PIPELINE_CONFIG_SHA256]`
- Dataset configuration SHA-256: `[DATASET_CONFIG_SHA256]`
- Tokenizer configuration SHA-256: `[TOKENIZER_CONFIG_SHA256]`
- Training configuration SHA-256: `[TRAIN_CONFIG_SHA256]`
- Model configuration SHA-256: `[MODEL_CONFIG_SHA256]`
- Shard manifest SHA-256: `[SHARD_MANIFEST_SHA256]`

## Training procedure

- Model preset: `[MODEL_PRESET]`
- Sequence length: `[SEQUENCE_LENGTH]`
- Training steps: `[TRAINING_STEPS]`
- Seed: `[SEED]`
- Hardware: `[HARDWARE]`

## Evaluation

`[EVALUATION_RESULTS]`

Do not present a smoke run as evidence of production model quality. Record the evaluation dataset, metrics, methodology, and limitations for any quality claim.

## Export integrity

- Final GGUF SHA-256: `[FINAL_GGUF_SHA256]`
- F16 GGUF SHA-256: `[F16_GGUF_SHA256]`
- llama.cpp converter revision: `[CONVERTER_REVISION]`
- Export manifest: `[EXPORT_MANIFEST]`

## Limitations and risks

`[LIMITATIONS_AND_RISKS]`

The model may reproduce errors, bias, unsafe instructions, or other defects present in its training data. The model card describes the artifact and its evidence; it does not constitute a safety certification.

## License

`[MODEL_LICENSE]`

Confirm that the model license is compatible with the training-data licenses, attribution requirements, and intended distribution before publishing the artifact.
