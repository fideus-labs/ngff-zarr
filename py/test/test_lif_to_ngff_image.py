# SPDX-FileCopyrightText: Copyright (c) Fideus Labs LLC
# SPDX-License-Identifier: MIT
"""Tests for LIF to NGFF image conversion using mocked liffile objects."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from ngff_zarr.lif_to_ngff_image import (
    _extract_scale_translation,
    _map_lif_dims_to_ngff,
    has_mosaic_dimension,
    lif_to_hcs_plate,
    lif_to_ngff_image,
)


@pytest.fixture
def mock_lif_image_xyz():
    """Create a mock LifImage with XYZ dimensions."""
    image = Mock()
    image.name = "test_xyz"
    image.path = "test_xyz"
    image.shape = (10, 128, 128)  # Z, Y, X
    image.dims = ("Z", "Y", "X")
    image.sizes = {"Z": 10, "Y": 128, "X": 128}
    image.dtype = np.uint8
    image.coords = {
        "Z": np.linspace(0, 9e-6, 10),
        "Y": np.linspace(0, 127e-6, 128),
        "X": np.linspace(0, 127e-6, 128),
    }
    # Return test data when asarray() is called
    image.asarray.return_value = np.random.randint(
        0, 255, (10, 128, 128), dtype=np.uint8
    )
    return image


@pytest.fixture
def mock_lif_image_xyztc():
    """Create a mock LifImage with XYZTC dimensions."""
    image = Mock()
    image.name = "test_xyztc"
    image.path = "test_xyztc"
    image.shape = (5, 10, 2, 128, 128)  # T, Z, C, Y, X
    image.dims = ("T", "Z", "C", "Y", "X")
    image.sizes = {"T": 5, "Z": 10, "C": 2, "Y": 128, "X": 128}
    image.dtype = np.uint16
    image.coords = {
        "T": np.linspace(0, 4.0, 5),  # 5 timepoints
        "Z": np.linspace(0, 9e-6, 10),
        "Y": np.linspace(0, 127e-6, 128),
        "X": np.linspace(0, 127e-6, 128),
    }
    image.asarray.return_value = np.random.randint(
        0, 65535, (5, 10, 2, 128, 128), dtype=np.uint16
    )
    return image


@pytest.fixture
def mock_lif_image_lambda():
    """Create a mock LifImage with emission wavelength (lambda) dimension."""
    image = Mock()
    image.name = "test_lambda"
    image.path = "test_lambda"
    image.shape = (9, 128, 128)  # λ, Y, X
    image.dims = ("λ", "Y", "X")
    image.sizes = {"λ": 9, "Y": 128, "X": 128}
    image.dtype = np.uint8
    image.coords = {
        "λ": np.linspace(400e-9, 700e-9, 9),  # wavelength in meters
        "Y": np.linspace(0, 127e-6, 128),
        "X": np.linspace(0, 127e-6, 128),
    }
    image.asarray.return_value = np.random.randint(
        0, 255, (9, 128, 128), dtype=np.uint8
    )
    return image


@pytest.fixture
def mock_lif_image_mosaic():
    """Create a mock LifImage with mosaic (M) dimension > 1."""
    image = Mock()
    image.name = "test_mosaic"
    image.path = "test_mosaic"
    image.shape = (4, 10, 128, 128)  # M, Z, Y, X
    image.dims = ("M", "Z", "Y", "X")
    image.sizes = {"M": 4, "Z": 10, "Y": 128, "X": 128}
    image.dtype = np.uint8
    image.coords = {
        "Z": np.linspace(0, 9e-6, 10),
        "Y": np.linspace(0, 127e-6, 128),
        "X": np.linspace(0, 127e-6, 128),
    }
    image.asarray.return_value = np.random.randint(
        0, 255, (4, 10, 128, 128), dtype=np.uint8
    )
    return image


@pytest.fixture
def mock_lif_image_multi_channel():
    """Create a mock LifImage with multiple channel-like dimensions."""
    image = Mock()
    image.name = "test_multi_channel"
    image.path = "test_multi_channel"
    # Z, C, λ, Y, X - both C and λ are channel-like
    image.shape = (10, 2, 3, 128, 128)
    image.dims = ("Z", "C", "λ", "Y", "X")
    image.sizes = {"Z": 10, "C": 2, "λ": 3, "Y": 128, "X": 128}
    image.dtype = np.uint8
    image.coords = {
        "Z": np.linspace(0, 9e-6, 10),
        "Y": np.linspace(0, 127e-6, 128),
        "X": np.linspace(0, 127e-6, 128),
    }
    image.asarray.return_value = np.random.randint(
        0, 255, (10, 2, 3, 128, 128), dtype=np.uint8
    )
    return image


class TestDimensionMapping:
    """Tests for LIF to NGFF dimension mapping."""

    def test_map_xyz_dims(self):
        """Test mapping of basic XYZ dimensions."""
        lif_dims = ("Z", "Y", "X")
        lif_shape = (10, 128, 128)

        ngff_dims, ngff_shape = _map_lif_dims_to_ngff(lif_dims, lif_shape)

        assert ngff_dims == ("z", "y", "x")
        assert ngff_shape == (10, 128, 128)

    def test_map_xyztc_dims(self):
        """Test mapping of XYZTC dimensions."""
        lif_dims = ("T", "Z", "C", "Y", "X")
        lif_shape = (5, 10, 2, 128, 128)

        ngff_dims, ngff_shape = _map_lif_dims_to_ngff(lif_dims, lif_shape)

        # C should be mapped to c and placed in appropriate position
        assert "t" in ngff_dims
        assert "z" in ngff_dims
        assert "c" in ngff_dims
        assert "y" in ngff_dims
        assert "x" in ngff_dims

    def test_map_lambda_to_channel(self):
        """Test that emission wavelength (λ) maps to channel (c)."""
        lif_dims = ("λ", "Y", "X")
        lif_shape = (9, 128, 128)

        ngff_dims, ngff_shape = _map_lif_dims_to_ngff(lif_dims, lif_shape)

        assert "c" in ngff_dims
        assert ngff_dims.count("c") == 1  # Only one channel dimension

    def test_flatten_multiple_channel_dims(self):
        """Test that multiple channel-like dims are flattened."""
        lif_dims = ("Z", "C", "λ", "Y", "X")
        lif_shape = (10, 2, 3, 128, 128)  # 2 channels * 3 wavelengths = 6 total

        ngff_dims, ngff_shape = _map_lif_dims_to_ngff(lif_dims, lif_shape)

        # Multiple channel dims should be flattened
        c_count = ngff_dims.count("c")
        assert c_count == 1  # Single flattened channel dimension
        # Total channels = 2 * 3 = 6
        c_idx = ngff_dims.index("c")
        assert ngff_shape[c_idx] == 6

    def test_mosaic_dimension_preserved(self):
        """Test that mosaic (M) dimension is preserved as 'm'."""
        lif_dims = ("M", "Z", "Y", "X")
        lif_shape = (4, 10, 128, 128)

        ngff_dims, ngff_shape = _map_lif_dims_to_ngff(lif_dims, lif_shape)

        assert "m" in ngff_dims


class TestScaleTranslationExtraction:
    """Tests for extracting scale and translation from LIF coordinates."""

    def test_extract_xyz_scale_translation(self, mock_lif_image_xyz):
        """Test extraction of scale and translation for XYZ image."""
        ngff_dims = ("z", "y", "x")

        scale, translation = _extract_scale_translation(mock_lif_image_xyz, ngff_dims)

        # Check scale is approximately 1 micrometer
        assert "z" in scale
        assert "y" in scale
        assert "x" in scale
        assert scale["z"] == pytest.approx(1e-6, rel=0.01)
        assert scale["y"] == pytest.approx(1e-6, rel=0.01)
        assert scale["x"] == pytest.approx(1e-6, rel=0.01)

        # Check translation (origin)
        assert "z" in translation
        assert "y" in translation
        assert "x" in translation
        assert translation["z"] == pytest.approx(0.0)
        assert translation["y"] == pytest.approx(0.0)
        assert translation["x"] == pytest.approx(0.0)


class TestLifToNgffImage:
    """Tests for the main lif_to_ngff_image conversion function."""

    def test_basic_xyz_conversion(self, mock_lif_image_xyz):
        """Test basic XYZ image conversion."""
        ngff_image = lif_to_ngff_image(mock_lif_image_xyz)

        assert ngff_image.name == "test_xyz"
        assert ngff_image.dims == ("z", "y", "x")
        assert ngff_image.data.shape == (10, 128, 128)
        assert ngff_image.data.dtype == np.uint8
        assert "z" in ngff_image.scale
        assert "y" in ngff_image.scale
        assert "x" in ngff_image.scale

    def test_xyztc_conversion(self, mock_lif_image_xyztc):
        """Test XYZTC image conversion."""
        ngff_image = lif_to_ngff_image(mock_lif_image_xyztc)

        assert ngff_image.name == "test_xyztc"
        assert "t" in ngff_image.dims
        assert "z" in ngff_image.dims
        assert "c" in ngff_image.dims
        assert "y" in ngff_image.dims
        assert "x" in ngff_image.dims
        assert ngff_image.data.dtype == np.uint16

    def test_custom_name(self, mock_lif_image_xyz):
        """Test that custom name is applied."""
        ngff_image = lif_to_ngff_image(mock_lif_image_xyz, name="custom_name")

        assert ngff_image.name == "custom_name"

    def test_lambda_to_channel(self, mock_lif_image_lambda):
        """Test that λ dimension is converted to channel."""
        ngff_image = lif_to_ngff_image(mock_lif_image_lambda)

        assert "c" in ngff_image.dims
        # λ dimension (9) should become channel
        c_idx = list(ngff_image.dims).index("c")
        assert ngff_image.data.shape[c_idx] == 9


class TestHasMosaicDimension:
    """Tests for has_mosaic_dimension function."""

    def test_has_mosaic_true(self, mock_lif_image_mosaic):
        """Test detection of mosaic dimension > 1."""
        assert has_mosaic_dimension(mock_lif_image_mosaic) is True

    def test_has_mosaic_false_no_m_dim(self, mock_lif_image_xyz):
        """Test no mosaic when M dimension absent."""
        assert has_mosaic_dimension(mock_lif_image_xyz) is False

    def test_has_mosaic_false_m_equals_1(self):
        """Test no mosaic when M dimension equals 1."""
        image = Mock()
        image.dims = ("M", "Z", "Y", "X")
        image.shape = (1, 10, 128, 128)  # M=1
        image.sizes = {"M": 1, "Z": 10, "Y": 128, "X": 128}

        assert has_mosaic_dimension(image) is False


class TestLifToHcsPlate:
    """Tests for lif_to_hcs_plate conversion."""

    def test_mosaic_to_hcs_plate(self, mock_lif_image_mosaic):
        """Test conversion of mosaic image to HCS plate."""
        plate, well_images = lif_to_hcs_plate(
            mock_lif_image_mosaic, plate_name="test_plate"
        )

        # Check plate structure
        assert plate.name == "test_plate"
        assert len(plate.rows) == 1  # Single row
        assert plate.rows[0].name == "A"
        assert len(plate.columns) == 4  # 4 mosaic positions
        assert len(plate.wells) == 4

        # Check well images
        assert len(well_images) == 4
        for row_name, col_name, field_idx, ngff_image in well_images:
            assert row_name == "A"
            assert field_idx == 0
            # Each position should have z, y, x dims (M removed)
            assert "m" not in ngff_image.dims
            assert "z" in ngff_image.dims
            assert "y" in ngff_image.dims
            assert "x" in ngff_image.dims

    def test_non_mosaic_raises_error(self, mock_lif_image_xyz):
        """Test that non-mosaic image raises ValueError."""
        with pytest.raises(ValueError, match="does not have a mosaic dimension"):
            lif_to_hcs_plate(mock_lif_image_xyz)


class TestLifFileToNgffImages:
    """Tests for lif_file_to_ngff_images function using mocked LifFile."""

    @patch("liffile.LifFile")
    def test_convert_all_series(self, mock_liffile_class):
        """Test converting all series from a LIF file."""
        # Setup mock
        mock_lif = MagicMock()
        mock_liffile_class.return_value.__enter__ = Mock(return_value=mock_lif)
        mock_liffile_class.return_value.__exit__ = Mock(return_value=False)

        # Create mock images
        mock_img1 = Mock()
        mock_img1.name = "Series1"
        mock_img1.path = "Series1"
        mock_img1.dims = ("Z", "Y", "X")
        mock_img1.shape = (10, 128, 128)
        mock_img1.dtype = np.uint8
        mock_img1.coords = {}
        mock_img1.asarray.return_value = np.zeros((10, 128, 128), dtype=np.uint8)

        mock_img2 = Mock()
        mock_img2.name = "Series2"
        mock_img2.path = "Series2"
        mock_img2.dims = ("Z", "Y", "X")
        mock_img2.shape = (5, 64, 64)
        mock_img2.dtype = np.uint8
        mock_img2.coords = {}
        mock_img2.asarray.return_value = np.zeros((5, 64, 64), dtype=np.uint8)

        mock_lif.images = [mock_img1, mock_img2]

        from ngff_zarr.lif_to_ngff_image import lif_file_to_ngff_images

        results = lif_file_to_ngff_images("test.lif", series=None)

        assert len(results) == 2
        assert results[0][0] == "Series1"
        assert results[1][0] == "Series2"

    @patch("liffile.LifFile")
    def test_convert_single_series_by_index(self, mock_liffile_class):
        """Test converting a single series by index."""
        mock_lif = MagicMock()
        mock_liffile_class.return_value.__enter__ = Mock(return_value=mock_lif)
        mock_liffile_class.return_value.__exit__ = Mock(return_value=False)

        mock_img1 = Mock()
        mock_img1.name = "Series1"
        mock_img1.path = "Series1"
        mock_img1.dims = ("Z", "Y", "X")
        mock_img1.shape = (10, 128, 128)
        mock_img1.dtype = np.uint8
        mock_img1.coords = {}
        mock_img1.asarray.return_value = np.zeros((10, 128, 128), dtype=np.uint8)

        mock_img2 = Mock()
        mock_img2.name = "Series2"
        mock_img2.path = "Series2"
        mock_img2.dims = ("Z", "Y", "X")
        mock_img2.shape = (5, 64, 64)
        mock_img2.dtype = np.uint8
        mock_img2.coords = {}
        mock_img2.asarray.return_value = np.zeros((5, 64, 64), dtype=np.uint8)

        mock_lif.images = [mock_img1, mock_img2]

        from ngff_zarr.lif_to_ngff_image import lif_file_to_ngff_images

        results = lif_file_to_ngff_images("test.lif", series=1)

        assert len(results) == 1
        assert results[0][0] == "Series2"

    @patch("liffile.LifFile")
    def test_convert_series_by_pattern(self, mock_liffile_class):
        """Test converting series by name pattern."""
        mock_lif = MagicMock()
        mock_liffile_class.return_value.__enter__ = Mock(return_value=mock_lif)
        mock_liffile_class.return_value.__exit__ = Mock(return_value=False)

        mock_img1 = Mock()
        mock_img1.name = "SHG_area_1"
        mock_img1.path = "SHG_area_1"
        mock_img1.dims = ("Z", "Y", "X")
        mock_img1.shape = (10, 128, 128)
        mock_img1.dtype = np.uint8
        mock_img1.coords = {}
        mock_img1.asarray.return_value = np.zeros((10, 128, 128), dtype=np.uint8)

        mock_img2 = Mock()
        mock_img2.name = "Overview"
        mock_img2.path = "Overview"
        mock_img2.dims = ("Z", "Y", "X")
        mock_img2.shape = (5, 64, 64)
        mock_img2.dtype = np.uint8
        mock_img2.coords = {}
        mock_img2.asarray.return_value = np.zeros((5, 64, 64), dtype=np.uint8)

        mock_img3 = Mock()
        mock_img3.name = "SHG_area_2"
        mock_img3.path = "SHG_area_2"
        mock_img3.dims = ("Z", "Y", "X")
        mock_img3.shape = (10, 128, 128)
        mock_img3.dtype = np.uint8
        mock_img3.coords = {}
        mock_img3.asarray.return_value = np.zeros((10, 128, 128), dtype=np.uint8)

        mock_lif.images = [mock_img1, mock_img2, mock_img3]

        from ngff_zarr.lif_to_ngff_image import lif_file_to_ngff_images

        results = lif_file_to_ngff_images("test.lif", series="*SHG*")

        assert len(results) == 2
        names = [r[0] for r in results]
        assert "SHG_area_1" in names
        assert "SHG_area_2" in names
        assert "Overview" not in names
