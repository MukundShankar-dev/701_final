from learned_filters.learned_filter import LearnedFilter


def test_learned_filter_train_and_query() -> None:
    positives = [
        "ACGTAC",
        "CGTACG",
        "TTTTTT",
        "AAAAAA",
        "CCCCCC",
        "GGGGGG",
        "ATATAT",
        "CGCGCG",
    ]
    lf = LearnedFilter(k=6, random_seed=0)
    metrics = lf.train(positives, negative_count=len(positives))

    assert "accuracy" in metrics
    assert all(lf.contains(k) for k in positives)
