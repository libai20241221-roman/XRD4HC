from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


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
    amp, cen, fwhm, eta, bg0, bg1 = popt
    y_fit = pseudo_voigt(x, *popt)
    y_bg = bg0 + bg1 * x
    area = float(np.trapz(y_fit - y_bg, x))

    return PeakResult(two_theta=float(cen), fwhm_deg=float(abs(fwhm)), area=area, height=float(amp))


def preprocess_intensity(intensity: np.ndarray, smooth: bool = True, window: int = 11, polyorder: int = 3) -> np.ndarray:
    y = np.asarray(intensity, dtype=float)
    if smooth and len(y) >= window and window % 2 == 1:
        y = savgol_filter(y, window_length=window, polyorder=min(polyorder, window - 1))
    baseline = np.percentile(y, 1)
    y = y - baseline
    return np.clip(y, a_min=0, a_max=None)


def scherrer_size_nm(lambda_nm: float, k: float, theta_deg: float, fwhm_deg: float, inst_fwhm_deg: float = 0.0) -> float:
    beta_deg_sq = max(fwhm_deg**2 - inst_fwhm_deg**2, 1e-9)
    beta_rad = np.deg2rad(np.sqrt(beta_deg_sq))
    theta_rad = np.deg2rad(theta_deg)
    return float((k * lambda_nm) / (beta_rad * np.cos(theta_rad)))


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
) -> XRDResult:
    y = preprocess_intensity(intensity, smooth=smooth)

    p002 = fit_peak(two_theta, y, range_002)
    p100 = fit_peak(two_theta, y, range_100)

    theta002 = p002.two_theta / 2.0
    d002_nm = float(lambda_nm / (2 * np.sin(np.deg2rad(theta002))))

    lc_nm = scherrer_size_nm(lambda_nm, k_lc, theta002, p002.fwhm_deg, inst_fwhm_deg)
    la_nm = scherrer_size_nm(lambda_nm, k_la, p100.two_theta / 2.0, p100.fwhm_deg, inst_fwhm_deg)

    return XRDResult(peak_002=p002, peak_100=p100, d002_nm=d002_nm, lc_nm=lc_nm, la_nm=la_nm)
