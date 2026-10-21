from __future__ import annotations

import os
from typing import Any


import cs336_data.data_utils
def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    return cs336_data.data_utils.extract_text_from_warc_bytes(html_bytes)


def run_identify_language(text: str) -> tuple[Any, float]:
    return cs336_data.data_utils.identify_language(text)


def run_mask_emails(text: str) -> tuple[str, int]:
    return cs336_data.data_utils.mask_emails(text)


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    return cs336_data.data_utils.mask_phone(text)


def run_mask_ips(text: str) -> tuple[str, int]:
    return cs336_data.data_utils.mask_ip(text)


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    return cs336_data.data_utils.get_is_nsfw(text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    return cs336_data.data_utils.get_is_hatespeech(text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    raise NotImplementedError


def run_gopher_quality_filter(text: str) -> bool:
    return cs336_data.data_utils.check_gopher_good(text)


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    return cs336_data.data_utils.exact_line_deduplication(input_files, output_directory)


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    return cs336_data.data_utils.fuzzy_document_deduplication(input_files, output_directory, num_hashes, num_bands, ngrams, jaccard_threshold)
