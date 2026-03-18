import numpy as np

from xrd_analysis import analyze_hard_carbon_xrd


def _gaussian(x, amp, cen, sigma):
    return amp * np.exp(-((x - cen) ** 2) / (2 * sigma**2))


def test_analyze_hard_carbon_xrd_synthetic():
    x = np.linspace(10, 70, 2500)
    y = (
        _gaussian(x, 1000, 25.0, 1.2)
        + _gaussian(x, 500, 43.0, 1.0)
        + np.random.default_rng(0).normal(0, 5, x.size)
        + 30
    )

    res = analyze_hard_carbon_xrd(x, y, smooth=True)

    assert 24.0 <= res.peak_002.two_theta <= 26.0
    assert 42.0 <= res.peak_100.two_theta <= 44.0
    assert 0.33 <= res.d002_nm <= 0.38
    assert res.lc_nm > 0
    assert res.la_nm > 0
