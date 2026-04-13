from bloom_filters.bloom_filter import BloomFilter


def test_bloom_no_false_negatives() -> None:
    keys = ["ACGTAC", "CGTACG", "TTTTTT", "AAAAAA"]
    bf = BloomFilter(expected_items=len(keys), false_positive_rate=1e-3)
    bf.build(keys)

    for key in keys:
        assert bf.contains(key)
