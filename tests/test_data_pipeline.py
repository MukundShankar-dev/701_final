from data.kmers import canonical_kmer, generate_kmers, reverse_complement


def test_reverse_complement_and_canonical() -> None:
    assert reverse_complement("ACGT") == "ACGT"
    assert canonical_kmer("TTAA") == "TTAA"


def test_generate_kmers_filters_invalid() -> None:
    seq = "ACGTNACGT"
    kmers = list(generate_kmers(seq, k=4, canonical=False))
    assert "CGTN" not in kmers
    assert "GTNA" not in kmers
    assert "TNAC" not in kmers
    assert len(kmers) > 0
