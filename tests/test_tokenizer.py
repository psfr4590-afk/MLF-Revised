import json

import pytest

from pipeline.tokenizer.train_tokenizer import BPETokenizerTrainer, _doc_stream


def test_doc_stream_reads_valid_documents_and_skips_invalid(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"text": "first document"}\n'
        '{"not_text": "ignored"}\n'
        'not valid json\n'
        '{"text": ""}\n'
        '{"text": "second document"}\n',
        encoding="utf-8",
    )

    assert list(_doc_stream(corpus, None)) == [
        "first document",
        "second document",
    ]


def test_doc_stream_sampling_is_deterministic(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(json.dumps({"text": f"document {i}"}) + "\n" for i in range(100)),
        encoding="utf-8",
    )

    first = list(_doc_stream(corpus, 10, seed=42))
    second = list(_doc_stream(corpus, 10, seed=42))

    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10


def test_trainer_defaults_and_configuration(tmp_path):
    trainer = BPETokenizerTrainer(
        {
            "vocab_size": 128,
            "min_frequency": 3,
            "special_tokens": ["<|pad|>", "<|unk|>"],
            "output_path": str(tmp_path / "tokenizer"),
            "train_on_sample": True,
            "sample_size": 25,
        }
    )

    assert trainer._vocab_size == 128
    assert trainer._min_freq == 3
    assert trainer._special == ["<|pad|>", "<|unk|>"]
    assert trainer._sample_size == 25
    assert trainer._output_path == (tmp_path / "tokenizer").resolve()


def test_encode_requires_tokenizer(tmp_path):
    trainer = BPETokenizerTrainer({"output_path": str(tmp_path)})

    with pytest.raises(ValueError, match="Pass a trained tokenizer instance"):
        trainer.encode("hello")


def test_load_requires_artifact(tmp_path):
    trainer = BPETokenizerTrainer({"output_path": str(tmp_path)})

    with pytest.raises(FileNotFoundError, match="No tokenizer at"):
        trainer.load()


@pytest.mark.skipif(
    not __import__("pipeline.tokenizer.train_tokenizer", fromlist=["HF_TOKENIZERS_AVAILABLE"]).HF_TOKENIZERS_AVAILABLE,
    reason="tokenizers library unavailable",
)
def test_train_creates_valid_artifacts_and_provenance(tmp_path):
    from pipeline.tokenizer.train_tokenizer import _sha256_file

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(
            json.dumps(
                {
                    "text": (
                        f"document {i} contains enough repeated vocabulary "
                        "for a deterministic tokenizer training test"
                    )
                }
            )
            + "\n"
            for i in range(40)
        ),
        encoding="utf-8",
    )

    output = tmp_path / "tokenizer"
    trainer = BPETokenizerTrainer(
        {
            "vocab_size": 300,
            "min_frequency": 1,
            "output_path": str(output),
        }
    )

    tokenizer = trainer.train(corpus)

    assert tokenizer.get_vocab_size() == 300

    tokenizer_json = output / "tokenizer.json"
    config_json = output / "tokenizer_config.json"
    specials_json = output / "special_tokens_map.json"

    assert tokenizer_json.is_file()
    assert config_json.is_file()
    assert specials_json.is_file()

    config = json.loads(config_json.read_text(encoding="utf-8"))
    specials = json.loads(specials_json.read_text(encoding="utf-8"))

    assert config["vocab_size"] == 300
    assert config["provenance"]["corpus_sha256"] == _sha256_file(corpus)
    assert config["provenance"]["vocab_size_requested"] == 300
    assert config["provenance"]["bytelevel_alphabet_size"] == 256
    assert len(config["provenance"]["bytelevel_alphabet_sha256"]) == 64

    for name in (
        "bos_token",
        "eos_token",
        "unk_token",
        "pad_token",
        "sep_token",
        "mask_token",
    ):
        assert name in config
        assert name in specials

    loaded = trainer.load()
    assert loaded.get_vocab_size() == 300
    assert trainer.encode("document vocabulary", loaded)
