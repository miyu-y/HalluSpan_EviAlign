"""Token-level hallucination detection and input-side evidence alignment.

This is the public, dependency-minimal version of
progress/0626_inference_npm_ft_token.py.  It masks one output subword at a
time, retrieves the most similar document tokens, and stores their offsets.
"""

from __future__ import annotations

import argparse
from copy import deepcopy

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from src.common import document_text, load_encoder_checkpoint, load_jsonl, mask_span, overlaps_hallucination, write_jsonl

DOC_PREFIX = "Document: "
TEXT_PREFIX = "Text: "


def response_tokens(tokenizer, text: str, max_length: int) -> list[dict]:
    encoded = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    rows = []
    for index, ((start, end), token_id) in enumerate(zip(encoded["offset_mapping"], encoded["input_ids"])):
        if start == end or token_id in tokenizer.all_special_ids:
            continue
        rows.append({"sequence_position": index, "char_start": start, "char_end": end, "text": text[start:end]})
    return rows


def pair_inputs(tokenizer, document: str, masked_text: str, max_length: int):
    encoded = tokenizer(
        DOC_PREFIX + document,
        TEXT_PREFIX + masked_text,
        truncation="only_first",
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    doc_positions, doc_indices = [], []
    for index, (sequence_id, (start, end), token_id) in enumerate(zip(
        encoded.sequence_ids(), encoded["offset_mapping"][0].tolist(), encoded["input_ids"][0].tolist()
    )):
        # Exclude the literal "Document: " prompt tokens as well as specials.
        if sequence_id != 0 or start == end or end <= len(DOC_PREFIX) or token_id in tokenizer.all_special_ids:
            continue
        doc_indices.append(index)
        doc_positions.append({"start": max(0, start - len(DOC_PREFIX)), "end": max(0, end - len(DOC_PREFIX))})
    return encoded, doc_indices, doc_positions


@torch.inference_mode()
def score_token(model, tokenizer, document: str, masked_text: str, max_length: int, top_k: int, device: torch.device):
    encoded, doc_indices, doc_positions = pair_inputs(tokenizer, document, masked_text, max_length)
    encoded = {key: value.to(device) for key, value in encoded.items() if key != "offset_mapping"}
    hidden = model(**encoded, return_dict=True).last_hidden_state[0]
    input_ids = encoded["input_ids"][0]
    mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False).flatten()
    if len(mask_positions) != 1:
        raise ValueError("Token inference requires exactly one mask token after tokenization.")
    if not doc_indices:
        return 0.0, [], [], []
    mask_vector = F.normalize(hidden[mask_positions[0]], dim=-1)
    doc_vectors = F.normalize(hidden[torch.tensor(doc_indices, device=device)], dim=-1)
    similarities = doc_vectors @ mask_vector
    k = min(top_k, len(doc_indices))
    values, indices = torch.topk(similarities, k=k)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[torch.tensor(doc_indices, device=device)].tolist())
    positions, token_strings = [], []
    for index in indices.tolist():
        position = dict(doc_positions[index])
        position["doc_token_index"] = index
        positions.append(position)
        token_strings.append(tokenizer.convert_tokens_to_string([all_tokens[index]]))
    return float(values[0]), [float(x) for x in values.tolist()], token_strings, positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Token-level EviAlign inference")
    parser.add_argument("--input_jsonl", required=True, help="RAGTruth-style JSONL containing response and source_info")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned checkpoint directory or pytorch_model.bin")
    parser.add_argument("--base_model", default="answerdotai/ModernBERT-large")
    parser.add_argument("--threshold", type=float, default=0.68, help="A token is hallucinated when max_score < threshold")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=8192)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = load_encoder_checkpoint(AutoModel.from_pretrained(args.base_model), args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    output = []
    for row in tqdm(load_jsonl(args.input_jsonl), desc="token inference"):
        item = deepcopy(row)
        response = item.get("response", "")
        document = document_text(item)
        labels = item.get("labels", [])
        tokens = []
        for token in response_tokens(tokenizer, response, args.max_length):
            masked = mask_span(response, token["char_start"], token["char_end"], tokenizer.mask_token, count=1)
            max_score, top_scores, predicted_tokens, positions = score_token(
                model, tokenizer, document, masked, args.max_length, args.top_k, device
            )
            token.update({
                "hallucinated": overlaps_hallucination(labels, token["char_start"], token["char_end"]),
                "max_score": max_score,
                "top_k_scores": top_scores,
                "predicted_tokens": predicted_tokens,
                "predicted_token_positions": positions,
                "predicted": int(max_score < args.threshold),
            })
            tokens.append(token)
        item["tokens"] = tokens
        item["threshold"] = args.threshold
        output.append(item)
    write_jsonl(args.output_jsonl, output)


if __name__ == "__main__":
    main()
