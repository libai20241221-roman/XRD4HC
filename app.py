import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from xrd_analysis import analyze_hard_carbon_xrd, preprocess_intensity
from xrd_io import read_xrd_table_from_upload

st.set_page_config(page_title="硬炭XRD快速分析", layout="wide")
st.title("硬炭 XRD 数据分析（La / Lc / d002）")
st.caption("支持上传 txt/csv/excel，自动识别2θ与强度列并拟合(002)/(100)峰。")

with st.sidebar:
    st.header("参数设置")
    lambda_nm = st.number_input("X射线波长 λ (nm)", value=0.15406, format="%.5f")
    k_lc = st.number_input("Scherrer常数 Kc (Lc)", value=0.89, format="%.3f")
    k_la = st.number_input("Scherrer常数 Ka (La)", value=1.84, format="%.3f")
    inst_fwhm = st.number_input("仪器展宽FWHM (°2θ)", value=0.0, format="%.4f")
    smooth = st.checkbox("Savitzky-Golay平滑", value=True)

uploaded = st.file_uploader("上传CSV/TXT/Excel，自动识别2θ与强度列", type=["csv", "txt", "dat", "xlsx", "xls"])

if uploaded:
    raw = uploaded.read()
    try:
        two_theta, intensity = read_xrd_table_from_upload(uploaded.name, raw)
    except Exception as e:
        st.error(f"文件解析失败：{e}")
        st.stop()

    try:
        result = analyze_hard_carbon_xrd(
            two_theta,
            intensity,
            lambda_nm=lambda_nm,
            k_lc=k_lc,
            k_la=k_la,
            inst_fwhm_deg=inst_fwhm,
            smooth=smooth,
        )
    except Exception as e:
        st.exception(e)
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("d002 (nm)", f"{result.d002_nm:.4f}")
    c2.metric("Lc (nm)", f"{result.lc_nm:.3f}")
    c3.metric("La (nm)", f"{result.la_nm:.3f}")

    st.subheader("峰拟合结果")
    st.write(
        pd.DataFrame(
            {
                "峰": ["(002)", "(100)"],
                "2θ中心(°)": [result.peak_002.two_theta, result.peak_100.two_theta],
                "FWHM(°2θ)": [result.peak_002.fwhm_deg, result.peak_100.fwhm_deg],
                "积分面积": [result.peak_002.area, result.peak_100.area],
            }
        )
    )

    y_proc = preprocess_intensity(intensity, smooth=smooth)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(two_theta, y_proc, lw=1.2, label="预处理后谱线")
    ax.axvline(result.peak_002.two_theta, color="tab:red", ls="--", label="(002)中心")
    ax.axvline(result.peak_100.two_theta, color="tab:green", ls="--", label="(100)中心")
    ax.set_xlabel("2θ (degree)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.legend()
    st.pyplot(fig)

    out_df = pd.DataFrame(
        {
            "d002_nm": [result.d002_nm],
            "Lc_nm": [result.lc_nm],
            "La_nm": [result.la_nm],
            "peak002_2theta_deg": [result.peak_002.two_theta],
            "peak002_fwhm_deg": [result.peak_002.fwhm_deg],
            "peak100_2theta_deg": [result.peak_100.two_theta],
            "peak100_fwhm_deg": [result.peak_100.fwhm_deg],
        }
    )
    st.download_button("下载结果CSV", out_df.to_csv(index=False).encode("utf-8"), "xrd_result.csv", "text/csv")
else:
    st.info("请先上传数据文件。")
