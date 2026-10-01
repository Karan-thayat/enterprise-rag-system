"""Fine-tunes the bi-encoder (all-MiniLM-L6-v2) on QASPER's training papers.

The reranker can only reorder the 20 candidates the first stage finds, so this
targets candidate recall. Loss: MultipleNegativesRankingLoss, i.e. cross-entropy
that ranks each question's evidence chunk above one hard negative from the same
paper (see eval/hard_negatives.py) and above every other passage in the batch.
"""

import math
import random
from collections.abc import Callable
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cross_entropy
from transformers import get_linear_schedule_with_warmup

from eval.hard_negatives import Example


def train_embedder(
    examples: list[Example],
    base_model: str,
    output_dir: Path,
    evaluate: Callable[[Path], float],
    learning_rate: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 64,
    scale: float = 20.0,
    warmup_ratio: float = 0.1,
    device: str = "cpu",
    seed: int = 13,
) -> list[dict]:
    """Trains with in-batch and hard negatives, scoring a checkpoint on dev after each epoch.

    ``evaluate`` receives a checkpoint directory and returns the dev score to maximise;
    the best checkpoint is kept in ``output_dir/best``.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = SentenceTransformer(base_model, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps = math.ceil(len(examples) / batch_size) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup_ratio * steps), steps)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # sentence-transformers >= 5.5 renamed tokenize() to preprocess() and adds non-tensor entries.
    prepare = getattr(model, "preprocess", model.tokenize)

    def embed(texts: list[str]) -> torch.Tensor:
        features = {k: v.to(device) if torch.is_tensor(v) else v for k, v in prepare(texts).items()}
        return model(features)["sentence_embedding"]  # the model's Normalize layer makes these unit length

    history: list[dict] = []
    best_score = -math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        order = examples[:]
        rng.shuffle(order)
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            positives = [rng.choice(example.positives) for example in batch]
            negatives = [rng.choice(example.negatives) for example in batch if example.negatives]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                questions = embed([example.question for example in batch])
                passages = embed(positives + negatives)
            # Row i's correct passage is column i; every other column is a negative.
            scores = scale * questions.float() @ passages.float().T
            loss = cross_entropy(scores, torch.arange(len(batch), device=scores.device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            losses.append(loss.item())

        model.eval()
        checkpoint = output_dir / f"epoch-{epoch}"
        model.save(str(checkpoint))
        score = evaluate(checkpoint)
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), "dev_score": score})
        if score > best_score:
            best_score = score
            model.save(str(output_dir / "best"))
    return history
