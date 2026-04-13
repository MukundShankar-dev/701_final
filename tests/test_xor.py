from xor_filters.xor_filter import XorFilter


def test_xor_wrapper_build_and_query() -> None:
    keys = ["ACGTAC", "CGTACG", "TTTTTT", "AAAAAA"]
    xf = XorFilter(fingerprint_bits=8, backend="fallback")
    xf.build(keys)

    assert all(xf.contains(k) for k in keys)
    stats = xf.stats()
    assert "backend" in stats["parameters"]
