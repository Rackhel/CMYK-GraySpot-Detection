"""LabelRefiner class (stub)."""


class LabelRefiner:
    def __init__(self, cfg):
        self.cfg = cfg

    def compute_priority_score(self, embeddings, labels, paths):
        raise NotImplementedError

    def compute_clustering_quality(self, embeddings, labels):
        raise NotImplementedError
