# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Unit tests for ngff_zarr.codecs module."""

import pytest
import numcodecs

from ngff_zarr.codecs import codec_from_name, get_available_codecs


class TestGetAvailableCodecs:
    def test_always_includes_base_codecs(self):
        codecs = get_available_codecs()
        assert "none" in codecs
        assert "gzip" in codecs
        assert "lz4" in codecs
        assert "zstd" in codecs

    def test_includes_blosc_variants(self):
        """numcodecs is installed in the test env so blosc should appear."""
        codecs = get_available_codecs()
        assert "blosc:lz4" in codecs
        assert "blosc:zstd" in codecs
        assert "blosc:zlib" in codecs
        assert "blosc:lz4hc" in codecs
        assert "blosc:blosclz" in codecs
        assert "blosc:snappy" in codecs
        assert "blosc:blosc" in codecs

    def test_returns_list(self):
        codecs = get_available_codecs()
        assert isinstance(codecs, list)
        assert all(isinstance(c, str) for c in codecs)


class TestCodecFromName:
    def test_none_returns_none(self):
        assert codec_from_name("none") is None

    # -- gzip -----------------------------------------------------------

    def test_gzip_default(self):
        c = codec_from_name("gzip")
        assert isinstance(c, numcodecs.GZip)
        assert c.level == 6

    def test_gzip_custom_level(self):
        c = codec_from_name("gzip", level=3)
        assert isinstance(c, numcodecs.GZip)
        assert c.level == 3

    # -- zstd -----------------------------------------------------------

    def test_zstd_default(self):
        c = codec_from_name("zstd")
        assert isinstance(c, numcodecs.Zstd)
        assert c.level == 3

    def test_zstd_custom_level(self):
        c = codec_from_name("zstd", level=10)
        assert isinstance(c, numcodecs.Zstd)
        assert c.level == 10

    # -- lz4 ------------------------------------------------------------

    def test_lz4(self):
        c = codec_from_name("lz4")
        assert isinstance(c, numcodecs.LZ4)

    # -- blosc ----------------------------------------------------------

    def test_blosc_bare(self):
        """'blosc' without a variant defaults to lz4."""
        c = codec_from_name("blosc")
        assert isinstance(c, numcodecs.Blosc)
        assert c.cname == "lz4"
        assert c.clevel == 5

    def test_blosc_zstd(self):
        c = codec_from_name("blosc:zstd")
        assert isinstance(c, numcodecs.Blosc)
        assert c.cname == "zstd"
        assert c.clevel == 5

    def test_blosc_lz4hc(self):
        c = codec_from_name("blosc:lz4hc")
        assert isinstance(c, numcodecs.Blosc)
        assert c.cname == "lz4hc"

    def test_blosc_custom_level(self):
        c = codec_from_name("blosc:zlib", level=9)
        assert isinstance(c, numcodecs.Blosc)
        assert c.cname == "zlib"
        assert c.clevel == 9

    # -- errors ---------------------------------------------------------

    def test_unknown_codec_raises(self):
        with pytest.raises(ValueError, match="Unknown codec"):
            codec_from_name("bogus")

    def test_error_lists_available(self):
        with pytest.raises(ValueError, match="gzip"):
            codec_from_name("invalid_codec")
