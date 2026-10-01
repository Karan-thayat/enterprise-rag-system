"""Fine-tunes the cross-encoder reranker on QASPER's training papers.

Training pairs come from the retrieval stage the reranker sees in the application:
for each training question, the hybrid (BM25 + dense, RRF) candidates that are not
evidence become hard negatives, and chunks of the evidence paragraphs become
positives. The loss is binary cross-entropy on the model's single relevance logit.

Checkpoints are compared on the dev split only; the test split is kept for the
final evaluation.
"""

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder
from torch.nn.functional import binary_cross_entropy_with_logits
from transformers import get_linear_schedule_with_warmup

from app.embeddings import Embedder
from eval.hard_negatives import Example
from eval.qasper import IndexedPaper, make_retriever


@dataclass(frozen=True)
class Pair:
    question: str
    passage: str
    label: float  # 1.0 relevant, 0.0 not


def mine_pairs(
    indexed_papers: list[IndexedPaper],
    embedder: Embedder,
    negatives_per_question: int = 7,
    max_positives: int = 3,
    candidates_k: int = 20,
    seed: int = 13,
) -> list[Pair]:
    """Builds (question, chunk, label) pairs with hard negatives from hybrid retrieval."""
    rng = random.Random(seed)
    pairs: list[Pair] = []
    for indexed in indexed_papers:
        retriever = make_retriever(indexed, embedder, reranker=None, hybrid=True, candidates_k=candidates_k)
        for question in indexed.paper.questions:
            if not question.evidence:
                continue
            positives = [chunk.text for chunk in indexed.chunks if chunk.page in question.evidence]
            rng.shuffle(positives)
            # Candidates arrive best-first, so the first non-evidence chunks are the hardest negatives.
            negatives = [
                chunk.text
                for chunk in retriever.retrieve(question.text, top_k=candidates_k)
                if chunk.page not in question.evidence
            ]
            pairs += [Pair(question.text, text, 1.0) for text in positives[:max_positives]]
            pairs += [Pair(question.text, text, 0.0) for text in negatives[:negatives_per_question]]
    return pairs


def train(
    pairs: list[Pair],
    base_model: str,
    output_dir: Path,
    evaluate: Callable[[Path], float],
    learning_rate: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 32,
    warmup_ratio: float = 0.1,
    max_length: int = 320,
    device: str = "cpu",
    seed: int = 13,
) -> list[dict]:
    """Trains, saving and scoring a checkpoint after every epoch.

    ``evaluate`` receives a checkpoint directory and returns the dev score to maximise.
    The best checkpoint is kept in ``output_dir/best``.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    model = CrossEncoder(base_model, device=device, max_length=max_length)
    network, tokenizer = model.model, model.tokenizer
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=0.01)
    steps = math.ceil(len(pairs) / batch_size) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup_ratio * steps), steps)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict] = []
    best_score = -math.inf
    for epoch in range(1, epochs + 1):
        network.train()
        order = list(range(len(pairs)))
        random.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = [pairs[i] for i in order[start : start + batch_size]]
            features = tokenizer(
                [pair.question for pair in batch],
                [pair.passage for pair in batch],
                padding=True,
                truncation="only_second",
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            labels = torch.tensor([pair.label for pair in batch], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = network(**features).logits.view(-1)
            loss = binary_cross_entropy_with_logits(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            losses.append(loss.item())

        network.eval()
        checkpoint = output_dir / f"epoch-{epoch}"
        model.save(str(checkpoint))
        score = evaluate(checkpoint)
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), "dev_score": score})
        if score > best_score:
            best_score = score
            model.save(str(output_dir / "best"))
    return history


def train_listwise(
    examples: list[Example],
    base_model: str,
    output_dir: Path,
    evaluate: Callable[[Path], float],
    learning_rate: float = 2e-5,
    epochs: int = 5,
    questions_per_batch: int = 8,
    negatives_per_question: int = 7,
    warmup_ratio: float = 0.1,
    max_length: int = 320,
    device: str = "cpu",
    seed: int = 13,
) -> list[dict]:
    """Listwise fine-tuning: softmax cross-entropy over each question's evidence chunk and
    its hard negatives, so the model learns to rank the answer first rather than to score
    each passage in isolation. Checkpoints are scored on dev after every epoch, like train()."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    usable = [example for example in examples if example.negatives]
    model = CrossEncoder(base_model, device=device, max_length=max_length)
    network, tokenizer = model.model, model.tokenizer
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=0.01)
    steps = math.ceil(len(usable) / questions_per_batch) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup_ratio * steps), steps)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict] = []
    best_score = -math.inf
    for epoch in range(1, epochs + 1):
        network.train()
        order = usable[:]
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), questions_per_batch):
            groups = []
            for example in order[start : start + questions_per_batch]:
                negatives = list(example.negatives)
                rng.shuffle(negatives)
                groups.append(
                    (example.question, [rng.choice(example.positives), *negatives[:negatives_per_question]])
                )
            features = tokenizer(
                [question for question, passages in groups for _ in passages],
                [passage for _, passages in groups for passage in passages],
                padding=True,
                truncation="only_second",
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = network(**features).logits.view(-1)
            # Each group's first passage is the evidence: -log softmax probability of it.
            group_logits = torch.split(logits.float(), [len(passages) for _, passages in groups])
            loss = torch.stack([torch.logsumexp(g, 0) - g[0] for g in group_logits]).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            losses.append(loss.item())

        network.eval()
        checkpoint = output_dir / f"epoch-{epoch}"
        model.save(str(checkpoint))
        score = evaluate(checkpoint)
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), "dev_score": score})
        if score > best_score:
            best_score = score
            model.save(str(output_dir / "best"))
    return history
