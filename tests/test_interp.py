import numpy as np
import pytest

from mverify.interp import PointInterpolator
from mverify.io.open import Grid


def plane(lat, lon):
    return 2.0 * lat + 0.5 * lon


def test_bilinear_is_exact_for_a_plane_with_descending_lat_and_0_360_lon():
    lat, lon = np.arange(60, 19.5, -0.5), np.arange(220, 300.01, 0.5)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    pts_lat, pts_lon = np.array([40.3, 21.1, 55.55]), np.array([-100.2, -75.0, -130.9])
    got = PointInterpolator(Grid(lat, lon), pts_lat, pts_lon)(plane(la, lo))
    assert got == pytest.approx(plane(pts_lat, pts_lon % 360))


def test_points_outside_a_regional_grid_are_nan():
    lat, lon = np.arange(20, 60.1, 1.0), np.arange(-130, -59.9, 1.0)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    got = PointInterpolator(Grid(lat, lon), np.array([40, 70, 40]), np.array([-100, -100, 10]))(la + lo)
    assert np.isfinite(got[0]) and np.isnan(got[1]) and np.isnan(got[2])


def test_global_grid_wraps_across_the_seam():
    lat, lon = np.arange(-90, 90.1, 1.0), np.arange(0, 360, 1.0)
    _, lo = np.meshgrid(lat, lon, indexing="ij")
    field = np.cos(np.deg2rad(lo))
    got = PointInterpolator(Grid(lat, lon), np.array([10.0]), np.array([359.5]))(field)
    assert got[0] == pytest.approx((np.cos(np.deg2rad(359)) + 1) / 2, rel=1e-6)


def test_curvilinear_idw_is_close_and_masks_far_points():
    y, x = np.meshgrid(np.arange(40), np.arange(50), indexing="ij")
    lat2, lon2 = 30 + 0.25 * y + 0.02 * x, -110 + 0.3 * x
    interp = PointInterpolator(Grid(lat2, lon2), np.array([35.0, 0.0]), np.array([-100.0, -100.0]))
    got = interp(plane(lat2, lon2))
    assert got[0] == pytest.approx(plane(35.0, -100.0), abs=0.3)
    assert np.isnan(got[1])
