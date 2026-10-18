"""Strict configuration validation with actionable errors."""
from __future__ import annotations

QUANTS = {"F16", "Q4_K_M", "Q5_K_M", "Q8_0", "ALL"}
PRESETS = {"85M", "117M", "360M"}
STAGES = {"crawl", "clean", "dedup", "weight", "tokenize", "shard", "train", "export"}
SECTION_KEYS = {
    "pipeline": {"name", "model_name", "output_dir", "scratch_dir", "seed", "resume", "log_level"},
    "stages": STAGES,
    "crawl": {"max_workers", "request_timeout", "retry_attempts", "retry_backoff", "respect_robots_txt", "politeness_delay", "max_pages_per_domain", "max_content_bytes", "max_redirects", "fail_on_malformed_jsonl", "dataset_groups_file", "dataset_profiles_file", "source_weights_file", "sources", "web", "github", "arxiv", "huggingface", "google", "local_files"},
    "clean": {"config_file"},
    "embed_dedup": {"mode", "model", "model_path", "allow_model_download", "batch_size", "similarity_threshold", "chunk_size", "index_type", "ivf_nlist"},
    "weight": {"config_file", "strategy"},
    "tokenizer": {"type", "vocab_size", "min_frequency", "special_tokens", "train_on_sample", "sample_size", "output_path"},
    "shard": {"sequence_length", "dtype", "shard_size_tokens", "output_dir", "val_fraction", "shuffle_docs", "seed"},
    "train": {"resume", "allow_cpu_training", "auto_size", "target_training_hours", "observed_tokens_per_sec", "model_preset", "vocab_size", "seq_len", "dropout", "lr_max", "lr_min", "weight_decay", "grad_clip", "batch_size", "grad_accum_steps", "total_steps", "warmup_steps", "eval_every_steps", "eval_batches", "checkpoint_every_steps", "keep_checkpoints", "shard_dir"},
    "export": {"format", "llamacpp_dir", "quant", "model_name"},
    "command_center": {"host", "port", "auto_seed_datasets"},
    "preflight": {"enabled", "warmup_steps", "benchmark_steps", "capability_discovery", "capability_benchmark_steps", "capability_max_model_params", "capability_max_context", "capability_model_probe_rounds", "capability_context_probe_rounds"},
}
NESTED_KEYS = {
    "sources": {"web", "github", "arxiv", "huggingface", "google", "local_files"},
    "web": {"seed_urls_file", "seed_urls", "follow_links", "max_depth", "stay_on_domain", "use_sitemap_discovery", "english_only"},
    "github": {"include_wikis", "include_issues", "max_issues_per_repo", "token_env"},
    "arxiv": {"fetch_full_text", "fetch_abstract_only", "enrich_citations"},
    "huggingface": {"token_env"},
    "google": {"api_key_env", "cx_env", "results_per_query", "max_pages_per_query", "seed_urls", "max_seed_urls"},
    "local_files": {"paths", "recursive", "extensions", "max_files", "max_file_bytes"},
}


def _strict_keys(section: str, value: dict) -> None:
    unknown = set(value) - SECTION_KEYS[section]
    if unknown:
        raise ValueError(f"Unsupported {section} configuration keys: {', '.join(sorted(unknown))}")


def _strict_nested(name: str, value: dict) -> None:
    unknown = set(value) - NESTED_KEYS[name]
    if unknown:
        raise ValueError(f"Unsupported {name} configuration keys: {', '.join(sorted(unknown))}")


def _bool(section: str, key: str, value) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{section}.{key} must be boolean")
    return value


def validate_config(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        raise ValueError("Pipeline configuration must be a YAML object")
    required = ("pipeline", "stages", "crawl", "clean", "embed_dedup", "weight", "tokenizer", "shard", "train", "export")
    for section in required:
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Missing or invalid configuration section: {section}")
    for section, value in cfg.items():
        if section.startswith("_"):
            continue
        if section not in SECTION_KEYS:
            raise ValueError(f"Unsupported top-level configuration section: {section}")
        if not isinstance(value, dict):
            raise ValueError(f"Configuration section must be an object: {section}")
        _strict_keys(section, value)
    stages = cfg["stages"]
    invalid_stage_values = [name for name, enabled in stages.items() if not isinstance(enabled, bool)]
    if invalid_stage_values:
        raise ValueError(f"Pipeline stage flags must be boolean: {', '.join(sorted(invalid_stage_values))}")
    for name in ("sources", "web", "github", "arxiv", "huggingface", "google", "local_files"):
        value = cfg["crawl"].get(name)
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError(f"crawl.{name} must be an object")
            _strict_nested(name, value)
    local_files = cfg["crawl"].get("local_files", {})
    if bool((cfg["crawl"].get("sources", {}) or {}).get("local_files")) and not (local_files.get("paths", []) if isinstance(local_files, dict) else []):
        raise ValueError("crawl.local_files.paths is required when crawl.sources.local_files is enabled")
    if local_files:
        paths = local_files.get("paths", [])
        if not isinstance(paths, list) or any(not isinstance(path, str) or not path.strip() for path in paths):
            raise ValueError("crawl.local_files.paths must be a list of non-empty path strings")
        extensions = local_files.get("extensions", [])
        if not isinstance(extensions, list) or any(not isinstance(ext, str) or not ext.strip() for ext in extensions):
            raise ValueError("crawl.local_files.extensions must be a list of extension strings")
        for key in ("max_files", "max_file_bytes"):
            value = local_files.get(key, 1000 if key == "max_files" else 5_000_000)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"crawl.local_files.{key} must be a positive integer")
        if "recursive" in local_files:
            _bool("crawl.local_files", "recursive", local_files["recursive"])
    tok, shard, train, export = cfg["tokenizer"], cfg["shard"], cfg["train"], cfg["export"]
    if "seed" in cfg["pipeline"] and (not isinstance(cfg["pipeline"]["seed"], int) or isinstance(cfg["pipeline"]["seed"], bool)):
        raise ValueError("pipeline.seed must be an integer")
    preflight = cfg.get("preflight", {})
    if "enabled" in preflight:
        _bool("preflight", "enabled", preflight["enabled"])
    if "capability_discovery" in preflight:
        _bool("preflight", "capability_discovery", preflight["capability_discovery"])
    for key in ("warmup_steps", "benchmark_steps", "capability_benchmark_steps", "capability_max_model_params", "capability_max_context", "capability_model_probe_rounds", "capability_context_probe_rounds"):
        if key in preflight:
            value = preflight[key]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"preflight.{key} must be an integer")
            if key in {"warmup_steps", "capability_model_probe_rounds", "capability_context_probe_rounds"} and value < 0:
                raise ValueError(f"preflight.{key} must be non-negative")
            if key in {"benchmark_steps", "capability_benchmark_steps", "capability_max_model_params", "capability_max_context"} and value <= 0:
                raise ValueError(f"preflight.{key} must be positive")
    if "resume" in cfg["pipeline"]:
        _bool("pipeline", "resume", cfg["pipeline"]["resume"])
    for key in ("train_on_sample",):
        if key in tok:
            _bool("tokenizer", key, tok[key])
    if "shuffle_docs" in shard:
        _bool("shard", "shuffle_docs", shard["shuffle_docs"])
    for key in ("resume", "allow_cpu_training", "auto_size"):
        if key in train:
            _bool("train", key, train[key])
    vocab = tok.get("vocab_size")
    if not isinstance(vocab, int) or isinstance(vocab, bool) or vocab <= 0:
        raise ValueError("tokenizer.vocab_size must be a positive integer")
    if vocab != train.get("vocab_size"):
        raise ValueError("tokenizer.vocab_size must equal train.vocab_size")
    seq = shard.get("sequence_length")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq <= 0:
        raise ValueError("shard.sequence_length must be a positive integer")
    if seq != train.get("seq_len"):
        raise ValueError("shard.sequence_length must equal train.seq_len")
    if vocab > 65536 and str(shard.get("dtype", "uint16")) == "uint16":
        raise ValueError("tokenizer.vocab_size > 65536 requires shard.dtype=uint32")
    shard_size = shard.get("shard_size_tokens")
    if not isinstance(shard_size, int) or shard_size < seq + 1:
        raise ValueError("shard.shard_size_tokens must be at least sequence_length + 1")
    val = shard.get("val_fraction", 0.0)
    if not isinstance(val, (int, float)) or isinstance(val, bool) or not 0.0 < val < 1.0:
        raise ValueError("shard.val_fraction must be between 0 and 1")
    steps = train.get("total_steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("train.total_steps must be a positive integer")
    for key in ("batch_size", "grad_accum_steps", "eval_every_steps", "eval_batches", "checkpoint_every_steps", "keep_checkpoints"):
        value = train.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"train.{key} must be a positive integer")
    preset = train.get("model_preset", "85M")
    if not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError(f"train.model_preset '{preset}' is unsupported")
    for key in ("target_training_hours", "observed_tokens_per_sec"):
        value = train.get(key)
        if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0):
            raise ValueError(f"train.{key} must be > 0 when configured")
    warmup = train.get("warmup_steps", 0)
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0 or warmup >= steps:
        raise ValueError("train.warmup_steps must be an integer smaller than train.total_steps")
    quant = export.get("quant", "F16")
    if not isinstance(quant, str) or quant.upper() not in QUANTS:
        raise ValueError(f"export.quant '{quant}' is unsupported")
    if not isinstance(export.get("llamacpp_dir", ""), str) or not export["llamacpp_dir"].strip():
        raise ValueError("export.llamacpp_dir must be configured")
