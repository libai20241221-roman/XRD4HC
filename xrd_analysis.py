from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve


@dataclass
class PeakResult:
    two_theta: float
    fwhm_deg: float
    area: float
    height: float


@dataclass
class XRDResult:
    peak_002: PeakResult
    peak_100: PeakResult
    d002_nm: float
    lc_nm: float
    la_nm: float


def pseudo_voigt(x: np.ndarray, amp: float, cen: float, fwhm: float, eta: float, bg0: float, bg1: float) -> np.ndarray:
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gaussian = np.exp(-((x - cen) ** 2) / (2 * sigma**2))
    lorentz = 1.0 / (1.0 + 4.0 * ((x - cen) / fwhm) ** 2)
    return amp * (eta * lorentz + (1 - eta) * gaussian) + bg0 + bg1 * x


def _estimate_fwhm(x: np.ndarray, y: np.ndarray, center_idx: int) -> float:
    h = y[center_idx]
    half = h / 2.0
    left = center_idx
    while left > 0 and y[left] > half:
        left -= 1
    right = center_idx
    while right < len(y) - 1 and y[right] > half:
        right += 1
    return max(0.05, x[right] - x[left])


def fit_peak(two_theta: np.ndarray, intensity: np.ndarray, fit_range: Tuple[float, float]) -> PeakResult:
    mask = (two_theta >= fit_range[0]) & (two_theta <= fit_range[1])
    x = two_theta[mask]
    y = intensity[mask]
    if x.size < 7:
        raise ValueError(f"拟合区间 {fit_range} 数据点不足")

    i_max = int(np.argmax(y))
    cen0 = float(x[i_max])
    amp0 = float(max(y) - np.percentile(y, 20))
    fwhm0 = _estimate_fwhm(x, y, i_max)
    p0 = [amp0, cen0, fwhm0, 0.5, float(np.min(y)), 0.0]

    bounds = (
        [0.0, fit_range[0], 0.01, 0.0, -np.inf, -np.inf],
        [np.inf, fit_range[1], 10.0, 1.0, np.inf, np.inf],
    )

    popt, _ = curve_fit(pseudo_voigt, x, y, p0=p0, bounds=bounds, maxfev=30000)
    amp, cen, fwhm, _eta, bg0, bg1 = popt
    y_fit = pseudo_voigt(x, *popt)
    y_bg = bg0 + bg1 * x
    area = float(np.trapz(y_fit - y_bg, x))

    return PeakResult(two_theta=float(cen), fwhm_deg=float(abs(fwhm)), area=area, height=float(amp))


def _rolling_min_baseline(y: np.ndarray, window: int = 51) -> np.ndarray:
    w = max(5, int(window))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    baseline = np.empty_like(y)
    for i in range(len(y)):
        baseline[i] = np.min(ypad[i : i + w])
    return baseline


def _poly_baseline(y: np.ndarray, degree: int = 2) -> np.ndarray:
    x = np.arange(len(y), dtype=float)
    deg = min(max(1, degree), 5)
    coef = np.polyfit(x, y, deg=deg)
    return np.polyval(coef, x)


def _asls_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01, niter: int = 10) -> np.ndarray:
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def preprocess_intensity(
    intensity: np.ndarray,
    smooth: bool = True,
    window: int = 11,
    polyorder: int = 3,
    baseline_mode: str = "percentile",
    smooth_mode: str = "savgol",
    ma_window: int = 7,
    baseline_poly_degree: int = 2,
    asls_lam: float = 1e5,
    asls_p: float = 0.01,
) -> np.ndarray:
    y = np.asarray(intensity, dtype=float)

    if smooth:
        if smooth_mode == "moving_average":
            w = max(3, int(ma_window))
            kernel = np.ones(w) / w
            y = np.convolve(y, kernel, mode="same")
        elif smooth_mode == "savgol" and len(y) >= window and window % 2 == 1:
            y = savgol_filter(y, window_length=window, polyorder=min(polyorder, window - 1))

    if baseline_mode == "none":
        y_corr = y
    elif baseline_mode == "rolling_min":
        baseline = _rolling_min_baseline(y, window=max(31, window * 5))
        y_corr = y - baseline
    elif baseline_mode == "poly":
        baseline = _poly_baseline(y, degree=baseline_poly_degree)
        y_corr = y - baseline
    elif baseline_mode == "asls":
        baseline = _asls_baseline(y, lam=asls_lam, p=asls_p)
        y_corr = y - baseline
    else:
        baseline = np.percentile(y, 1)
        y_corr = y - baseline

    return np.clip(y_corr, a_min=0, a_max=None)


def scherrer_size_nm(lambda_nm: float, k: float, theta_deg: float, fwhm_deg: float, inst_fwhm_deg: float = 0.0) -> float:
    beta_deg_sq = max(fwhm_deg**2 - inst_fwhm_deg**2, 1e-9)
    beta_rad = np.deg2rad(np.sqrt(beta_deg_sq))
    theta_rad = np.deg2rad(theta_deg)
    return float((k * lambda_nm) / (beta_rad * np.cos(theta_rad)))


def suggest_peak_ranges(two_theta: np.ndarray, intensity: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    def _auto_range(default_range: Tuple[float, float], half_width: float) -> Tuple[float, float]:
        m = (two_theta >= default_range[0]) & (two_theta <= default_range[1])
        if m.sum() < 7:
            return default_range
        x = two_theta[m]
        y = intensity[m]
        center = float(x[int(np.argmax(y))])
        lo = max(default_range[0], center - half_width)
        hi = min(default_range[1], center + half_width)
        if hi - lo < 2.0:
            return default_range
        return (lo, hi)

    return _auto_range((20.0, 32.0), 3.5), _auto_range((38.0, 52.0), 4.0)


def analyze_hard_carbon_xrd(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    lambda_nm: float = 0.15406,
    k_lc: float = 0.89,
    k_la: float = 1.84,
    range_002: Tuple[float, float] = (20.0, 32.0),
    range_100: Tuple[float, float] = (38.0, 52.0),
    inst_fwhm_deg: float = 0.0,
    smooth: bool = True,
    baseline_mode: str = "percentile",
    smooth_mode: str = "savgol",
    ma_window: int = 7,
    baseline_poly_degree: int = 2,
    asls_lam: float = 1e5,
    asls_p: float = 0.01,
) -> XRDResult:
    y = preprocess_intensity(
        intensity,
        smooth=smooth,
        baseline_mode=baseline_mode,
        smooth_mode=smooth_mode,
        ma_window=ma_window,
        baseline_poly_degree=baseline_poly_degree,
        asls_lam=asls_lam,
        asls_p=asls_p,
    )

    p002 = fit_peak(two_theta, y, range_002)
    p100 = fit_peak(two_theta, y, range_100)

    theta002 = p002.two_theta / 2.0
    d002_nm = float(lambda_nm / (2 * np.sin(np.deg2rad(theta002))))

    lc_nm = scherrer_size_nm(lambda_nm, k_lc, theta002, p002.fwhm_deg, inst_fwhm_deg)
    la_nm = scherrer_size_nm(lambda_nm, k_la, p100.two_theta / 2.0, p100.fwhm_deg, inst_fwhm_deg)

    return XRDResult(peak_002=p002, peak_100=p100, d002_nm=d002_nm, lc_nm=lc_nm, la_nm=la_nm)
