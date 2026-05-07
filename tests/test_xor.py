from xor_filters.xor_filter import XorFilter


def test_xor_wrapper_build_and_query() -> None:
    keys = ["ACGTAC", "CGTACG", "TTTTTT", "AAAAAA"]
    xf = XorFilter(fingerprint_bits=8, backend="python")
    xf.build(keys)

    assert all(xf.contains(k) for k in keys)
    assert xf.contains("acgtac")
    stats = xf.stats()
    assert stats["parameters"]["backend"] == "python:xor"


def test_xor_save_load_preserves_membership(tmp_path) -> None:
    keys = ["ACGTAC", "CGTACG", "TTTTTT", "AAAAAA"]
    xf = XorFilter(fingerprint_bits=8, backend="python")
    xf.build(keys)

    path = tmp_path / "xor.json"
    xf.save(path)

    loaded = XorFilter.load(path)
    assert all(loaded.contains(k) for k in keys)
    assert loaded.stats()["parameters"]["backend"] == "python:xor"
