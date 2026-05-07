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


def test_cuckoo_normalizes_dna_case() -> None:
    cf = CuckooFilter(capacity=8, bucket_size=2, fingerprint_bits=8)
    cf.build(["ACGTAC"])

    assert cf.contains("acgtac")
    assert cf.delete("acgtac")


def test_cuckoo_failed_insert_rolls_back_evictions() -> None:
    cf = CuckooFilter(
        capacity=1,
        bucket_size=1,
        fingerprint_bits=8,
        max_relocations=1,
        random_seed=0,
    )

    inserted = 0
    for i in range(100):
        before = [bucket.copy() for bucket in cf._buckets]
        ok = cf.insert(f"KEY{i}")
        if ok:
            inserted += 1
            continue

        assert inserted > 0
        assert cf._buckets == before
        break
    else:
        raise AssertionError("Expected tiny Cuckoo filter to reject an insertion")
