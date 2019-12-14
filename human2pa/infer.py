from pathlib import Path

from flair.models import SequenceTagger, TextClassifier
from flair.embeddings import FlairEmbeddings, StackedEmbeddings, TokenEmbeddings


def make_embeddings(path: Path, prefix: str) -> TokenEmbeddings :
    forward = path / f"{prefix}-forward" / 'best-lm.pt'
    backward = path / f"{prefix}-backward" / 'best-lm.pt'
    embedding_types: List[TokenEmbeddings] = [
        FlairEmbeddings(forward),
        FlairEmbeddings(backward),
    ]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
    return embeddings


def load_cls_model(path: Path) -> TextClassifier:
    return TextClassifier.load(str(path / 'best-model.pt'))


def load_tag_model(path: Path) -> SequenceTagger:
    return SequenceTagger.load(str(path / 'best-model.pt'))
