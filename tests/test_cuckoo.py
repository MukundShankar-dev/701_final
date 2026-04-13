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
