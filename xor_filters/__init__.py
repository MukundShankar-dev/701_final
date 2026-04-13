"""XOR filter package for AMQ benchmarking."""

from xor_filters.builder import build_xor_filter, build_xor_filter_from_kmer_file
from xor_filters.xor_filter import XorFilter

__all__ = ["XorFilter", "build_xor_filter", "build_xor_filter_from_kmer_file"]
