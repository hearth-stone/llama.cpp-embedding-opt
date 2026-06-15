#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mteb
import numpy as np
import requests
from tqdm.auto import tqdm

try:
    from mteb.models import ModelMeta
except Exception:
    ModelMeta = None

try:
    from mteb.types import PromptType
except Exception:
    PromptType = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


BGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
TEXT_KEYS = (
    "text",
    "query",
    "body",
    "sentence",
    "sentence1",
    "sentence2",
    "premise",
    "hypothesis",
    "title",
)


class LlamaCppEmbeddingWrapper:
    def __init__(
        self,
        base_url: str,
        model: str,
        batch_size: int,
        timeout: int,
        add_bge_query_instruction: bool,
        show_progress: bool,
        tokenizer_path: str | None,
        max_tokens: int | None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.add_bge_query_instruction = add_bge_query_instruction
        self.show_progress = show_progress
        self.max_tokens = max_tokens
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self._mteb_model_meta = self._make_model_meta(max_tokens, add_bge_query_instruction)

    @property
    def mteb_model_meta(self) -> Any:
        return self._mteb_model_meta

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any = None,
        hf_split: str | None = None,
        hf_subset: str | None = None,
        prompt_type: Any = None,
        batch_size: int | None = None,
        **_: Any,
    ) -> np.ndarray:
        actual_batch_size = batch_size or self.batch_size
        embeddings: list[list[float]] = []
        progress = tqdm(
            desc=self._progress_desc(task_metadata, hf_split, hf_subset),
            unit="batch",
            disable=not self.show_progress,
        )

        try:
            for texts in self._iter_text_batches(inputs, actual_batch_size):
                if self._is_query(prompt_type):
                    texts = [BGE_ZH_QUERY_INSTRUCTION + text for text in texts]
                texts = self._truncate_texts(texts)
                embeddings.extend(self._embed_batch(texts))
                progress.update(1)
        finally:
            progress.close()

        return np.asarray(embeddings, dtype=np.float32)

    def similarity(self, embeddings1: Any, embeddings2: Any) -> np.ndarray:
        lhs = self._normalize(np.asarray(embeddings1, dtype=np.float32))
        rhs = self._normalize(np.asarray(embeddings2, dtype=np.float32))
        return lhs @ rhs.T

    def similarity_pairwise(self, embeddings1: Any, embeddings2: Any) -> np.ndarray:
        lhs = self._normalize(np.asarray(embeddings1, dtype=np.float32))
        rhs = self._normalize(np.asarray(embeddings2, dtype=np.float32))
        return np.sum(lhs * rhs, axis=-1)

    def _embed_batch(self, texts: list[Any]) -> list[list[float]]:
        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer no-key",
            },
            json={
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                "embedding request failed: "
                f"status={response.status_code} url={response.url} "
                f"batch_size={len(texts)} response={response.text[:1000]!r}"
            )

        items = response.json()["data"]
        items = sorted(items, key=lambda item: item["index"])
        return [item["embedding"] for item in items]

    def _iter_text_batches(self, inputs: Any, batch_size: int):
        if isinstance(inputs, dict):
            yield from self._split_texts(self._texts_from_batch(inputs), batch_size)
            return

        if isinstance(inputs, (list, tuple)):
            if not inputs:
                return
            if all(isinstance(item, str) for item in inputs):
                yield from self._split_texts(list(inputs), batch_size)
                return
            if all(isinstance(item, dict) for item in inputs):
                texts = [self._text_from_row(item) for item in inputs]
                yield from self._split_texts(texts, batch_size)
                return

        for batch in inputs:
            yield from self._split_texts(self._texts_from_batch(batch), batch_size)

    def _texts_from_batch(self, batch: Any) -> list[str]:
        if isinstance(batch, dict):
            for key in TEXT_KEYS:
                if key in batch:
                    value = batch[key]
                    if isinstance(value, (list, tuple)):
                        return [str(text) for text in value]
                    return [str(value)]
        if isinstance(batch, (list, tuple)):
            if all(isinstance(item, str) for item in batch):
                return [str(item) for item in batch]
            if all(isinstance(item, dict) for item in batch):
                return [self._text_from_row(item) for item in batch]
        raise TypeError(f"Unsupported MTEB input batch: {type(batch)}")

    def _text_from_row(self, row: dict[str, Any]) -> str:
        parts = [str(row[key]) for key in TEXT_KEYS if key in row and row[key] is not None]
        if parts:
            return " ".join(parts)
        raise TypeError(f"Unsupported MTEB input row keys: {sorted(row)}")

    def _split_texts(self, texts: list[str], batch_size: int):
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]

    def _is_query(self, prompt_type: Any) -> bool:
        if not self.add_bge_query_instruction:
            return False
        if PromptType is not None and prompt_type == PromptType.query:
            return True
        return str(prompt_type).lower().endswith("query")

    def _load_tokenizer(self, tokenizer_path: str | None) -> Any:
        if tokenizer_path is None:
            return None
        if AutoTokenizer is None:
            raise RuntimeError("transformers is required when --tokenizer is set")
        return AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    def _truncate_texts(self, texts: list[str]) -> list[Any]:
        if self.max_tokens is None:
            return texts
        if self.tokenizer is None:
            return self._truncate_texts_with_server_tokenizer(texts)

        truncated = []
        for text in texts:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_tokens,
            )
            truncated.append(
                self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        return truncated

    def _truncate_texts_with_server_tokenizer(self, texts: list[str]) -> list[list[int]]:
        truncated = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/tokenize",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer no-key",
                },
                json={
                    "content": text,
                    "add_special": True,
                    "parse_special": True,
                },
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    "tokenize request failed: "
                    f"status={response.status_code} url={response.url} "
                    f"response={response.text[:1000]!r}"
                )
            token_ids = response.json()["tokens"]
            truncated.append(token_ids[: self.max_tokens])
        return truncated

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    def _progress_desc(
        self,
        task_metadata: Any,
        hf_split: str | None,
        hf_subset: str | None,
    ) -> str:
        task_name = getattr(task_metadata, "name", None) or "mteb"
        parts = [str(task_name)]
        if hf_subset:
            parts.append(str(hf_subset))
        if hf_split:
            parts.append(str(hf_split))
        return "/".join(parts)

    def _make_model_meta(self, max_tokens: int | None, add_bge_query_instruction: bool) -> Any:
        if ModelMeta is None:
            return None
        return ModelMeta.create_empty(
            overwrites={
                "name": "local/llamacpp-bge-small-zh-v1.5",
                "framework": ["GGUF"],
                "max_tokens": max_tokens or 512,
                "embed_dim": 512,
                "use_instructions": add_bge_query_instruction,
            }
        )


def configure_quick_tasks(
    tasks: list[Any],
    *,
    quick: bool,
    samples_per_label: int | None,
    n_experiments: int | None,
    max_eval_samples: int | None,
) -> None:
    if quick:
        samples_per_label = samples_per_label or 1
        n_experiments = n_experiments or 1
        max_eval_samples = max_eval_samples or 500

    for task in tasks:
        if samples_per_label is not None and hasattr(task, "samples_per_label"):
            task.samples_per_label = samples_per_label
        if n_experiments is not None and hasattr(task, "n_experiments"):
            task.n_experiments = n_experiments
        if max_eval_samples is not None:
            truncate_eval_splits(task, max_eval_samples)


def truncate_eval_splits(task: Any, max_eval_samples: int) -> None:
    task.load_data()
    dataset = task.dataset
    if dataset is None:
        return

    for data_split in iter_dataset_dicts(dataset):
        for split in ("test", "validation", "dev"):
            if split in data_split:
                data_split[split] = truncate_value(data_split[split], max_eval_samples)


def truncate_value(value: Any, max_eval_samples: int) -> Any:
    if hasattr(value, "select"):
        return value.select(range(min(max_eval_samples, len(value))))

    if isinstance(value, list):
        return value[:max_eval_samples]

    if not isinstance(value, dict):
        return value

    if isinstance(value.get("queries"), dict):
        selected_query_ids = set(list(value["queries"].keys())[:max_eval_samples])
        truncated = dict(value)
        truncated["queries"] = {
            query_id: query
            for query_id, query in value["queries"].items()
            if query_id in selected_query_ids
        }
        for key in ("qrels", "relevant_docs", "scores"):
            if isinstance(value.get(key), dict):
                truncated[key] = {
                    query_id: docs
                    for query_id, docs in value[key].items()
                    if query_id in selected_query_ids
                }
        return truncated

    if value and all(not isinstance(item, (dict, list, tuple)) for item in value.values()):
        return dict(list(value.items())[:max_eval_samples])

    return value


def iter_dataset_dicts(dataset: Any):
    if hasattr(dataset, "keys") and any(split in dataset for split in ("train", "test", "validation", "dev")):
        yield dataset
        return

    if hasattr(dataset, "values"):
        for value in dataset.values():
            if hasattr(value, "keys"):
                yield value


def get_tasks(args: argparse.Namespace) -> list[Any]:
    kwargs: dict[str, Any] = {}
    if args.tasks:
        kwargs["tasks"] = args.tasks
    if args.languages:
        kwargs["languages"] = args.languages

    try:
        return mteb.get_tasks(**kwargs)
    except TypeError:
        if args.tasks:
            return mteb.get_tasks(tasks=args.tasks)
        return mteb.get_tasks(languages=args.languages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="bge")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--output-folder", default="results/llamacpp-bge")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--languages", nargs="+")
    parser.add_argument("--no-bge-query-instruction", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--samples-per-label", type=int)
    parser.add_argument("--n-experiments", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.all_tasks and not args.tasks:
        raise SystemExit("Pass --tasks TASK [TASK ...] or --all-tasks.")

    model = LlamaCppEmbeddingWrapper(
        base_url=args.base_url,
        model=args.model,
        batch_size=args.batch_size,
        timeout=args.timeout,
        add_bge_query_instruction=not args.no_bge_query_instruction,
        show_progress=not args.no_progress,
        tokenizer_path=args.tokenizer,
        max_tokens=args.max_tokens,
    )

    tasks = mteb.get_tasks(languages=args.languages) if args.all_tasks else get_tasks(args)
    configure_quick_tasks(
        tasks,
        quick=args.quick,
        samples_per_label=args.samples_per_label,
        n_experiments=args.n_experiments,
        max_eval_samples=args.max_eval_samples,
    )
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(
        model,
        output_folder=Path(args.output_folder),
        encode_kwargs={"batch_size": args.batch_size},
    )


if __name__ == "__main__":
    main()
