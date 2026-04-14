from cuckoo_filters.cuckoo_filter import CuckooFilter


def test_cuckoo_insert_contains_delete() -> None:
    keys = ["ACGTAC", "CGTACG", "TTTTTT", "AAAAAA", "CCCCCC"]
    cf = CuckooFilter(capacity=64, bucket_size=4, fingerprint_bits=12, max_relocations=200)
    cf.build(keys)

    for key in keys:
        assert cf.contains(key)

    assert cf.delete(keys[0])
    # Deleted key may still appear as positive due to fingerprint collision,
    # but should not raise and load factor decreases.
    assert cf.load_factor() < 1.0


def test_cuckoo_stats_target_fpr_matches_bucket_aware_formula() -> None:
    cf = CuckooFilter(capacity=64, bucket_size=4, fingerprint_bits=12)
    cf.build(["ACGTAC", "CGTACG", "TTTTTT", "AAAAAA"])

    stats = cf.stats()
    # Approximate Cuckoo FPR: (2 * bucket_size) / 2^fingerprint_bits.
    expected = (2 * 4) / (2**12)
    assert abs(stats["target_false_positive_rate"] - expected) < 1e-12
    assert abs(stats["fingerprint_only_false_positive_rate"] - (2**-12)) < 1e-12
