import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from xrd_analysis import analyze_hard_carbon_xrd, preprocess_intensity, suggest_peak_ranges
from xrd_io import read_xrd_table_from_upload

st.set_page_config(page_title="硬炭XRD快速分析", layout="wide")
st.title("🧪 硬炭 XRD 数据分析（La / Lc / d002）")
st.caption("支持批量上传 txt/csv/excel，自动识别2θ与强度列；支持一键自动与Origin风格手动分析。")

with st.sidebar:
    st.header("分析模式")
    mode = st.radio("选择模式", ["一键快速自动分析", "手动高级分析"], index=0)

    st.header("参数设置")
    lambda_nm = st.number_input("X射线波长 λ (nm)", value=0.15406, format="%.5f")
    k_lc = st.number_input("Scherrer常数 Kc (Lc)", value=0.89, format="%.3f")
    k_la = st.number_input("Scherrer常数 Ka (La)", value=1.84, format="%.3f")
    inst_fwhm = st.number_input("仪器展宽FWHM (°2θ)", value=0.0, format="%.4f")

    st.header("平滑设置")
    smooth = st.checkbox("启用平滑", value=True)
    smooth_mode = st.selectbox("平滑模式", ["savgol", "moving_average", "none"], format_func=lambda x: {"savgol": "Savitzky-Golay", "moving_average": "移动平均", "none": "不平滑"}[x])
    sg_window = st.number_input("S-G窗口(奇数)", min_value=5, max_value=101, value=11, step=2)
    sg_poly = st.number_input("S-G多项式阶数", min_value=1, max_value=7, value=3)
    ma_window = st.number_input("移动平均窗口", min_value=3, max_value=101, value=7)

    st.header("背景处理")
    baseline_mode = st.selectbox(
        "背景模式",
        options=["percentile", "rolling_min", "poly", "asls", "none"],
        format_func=lambda x: {
            "percentile": "分位数扣背景(默认)",
            "rolling_min": "滚动最小值",
            "poly": "多项式背景",
            "asls": "AsLS 基线",
            "none": "不扣背景",
        }[x],
    )
    poly_deg = st.number_input("多项式阶数", min_value=1, max_value=5, value=2)
    asls_lam = st.number_input("AsLS λ", min_value=1e2, max_value=1e8, value=1e5, format="%.0f")
    asls_p = st.number_input("AsLS p", min_value=0.0001, max_value=0.5, value=0.01, format="%.4f")

uploaded_files = st.file_uploader(
    "上传CSV/TXT/Excel（可多文件）",
    type=["csv", "txt", "dat", "xlsx", "xls"],
    accept_multiple_files=True,
)

if uploaded_files:
    results_rows = []
    tabs = st.tabs([f"文件{i+1}: {f.name}" for i, f in enumerate(uploaded_files)])

    for tab, up in zip(tabs, uploaded_files):
        with tab:
            raw = up.read()
            try:
                two_theta, intensity = read_xrd_table_from_upload(up.name, raw)
            except Exception as e:
                st.error(f"文件解析失败：{e}")
                continue

            if mode == "一键快速自动分析":
                range_002, range_100 = suggest_peak_ranges(
                    two_theta,
                    preprocess_intensity(
                        intensity,
                        smooth=smooth,
                        window=int(sg_window),
                        polyorder=int(sg_poly),
                        baseline_mode=baseline_mode,
                        smooth_mode=("none" if not smooth else smooth_mode),
                        ma_window=int(ma_window),
                        baseline_poly_degree=int(poly_deg),
                        asls_lam=float(asls_lam),
                        asls_p=float(asls_p),
                    ),
                )
                st.info(f"自动拟合区间：(002) {range_002[0]:.2f}–{range_002[1]:.2f}°, (100) {range_100[0]:.2f}–{range_100[1]:.2f}°")
            else:
                st.subheader("手动设置峰拟合区间")
                c1, c2 = st.columns(2)
                with c1:
                    r002_l = st.number_input(f"(002) 左边界 [{up.name}]", value=20.0, format="%.2f", key=f"r002l_{up.name}")
                    r002_r = st.number_input(f"(002) 右边界 [{up.name}]", value=32.0, format="%.2f", key=f"r002r_{up.name}")
                with c2:
                    r100_l = st.number_input(f"(100) 左边界 [{up.name}]", value=38.0, format="%.2f", key=f"r100l_{up.name}")
                    r100_r = st.number_input(f"(100) 右边界 [{up.name}]", value=52.0, format="%.2f", key=f"r100r_{up.name}")
                range_002 = (float(r002_l), float(r002_r))
                range_100 = (float(r100_l), float(r100_r))

            try:
                result = analyze_hard_carbon_xrd(
                    two_theta,
                    intensity,
                    lambda_nm=lambda_nm,
                    k_lc=k_lc,
                    k_la=k_la,
                    inst_fwhm_deg=inst_fwhm,
                    smooth=smooth,
                    range_002=range_002,
                    range_100=range_100,
                    baseline_mode=baseline_mode,
                    smooth_mode=("none" if not smooth else smooth_mode),
                    ma_window=int(ma_window),
                    baseline_poly_degree=int(poly_deg),
                    asls_lam=float(asls_lam),
                    asls_p=float(asls_p),
                )
            except Exception as e:
                st.exception(e)
                continue

            k1, k2, k3 = st.columns(3)
            k1.metric("d002 (nm)", f"{result.d002_nm:.4f}")
            k2.metric("Lc (nm)", f"{result.lc_nm:.3f}")
            k3.metric("La (nm)", f"{result.la_nm:.3f}")

            with st.expander("查看峰拟合参数", expanded=True):
                st.dataframe(
                    pd.DataFrame(
                        {
                            "峰": ["(002)", "(100)"],
                            "区间(°2θ)": [f"{range_002[0]:.2f}-{range_002[1]:.2f}", f"{range_100[0]:.2f}-{range_100[1]:.2f}"],
                            "2θ中心(°)": [result.peak_002.two_theta, result.peak_100.two_theta],
                            "FWHM(°2θ)": [result.peak_002.fwhm_deg, result.peak_100.fwhm_deg],
                            "积分面积": [result.peak_002.area, result.peak_100.area],
                        }
                    ),
                    use_container_width=True,
                )

            y_proc = preprocess_intensity(
                intensity,
                smooth=smooth,
                window=int(sg_window),
                polyorder=int(sg_poly),
                baseline_mode=baseline_mode,
                smooth_mode=("none" if not smooth else smooth_mode),
                ma_window=int(ma_window),
                baseline_poly_degree=int(poly_deg),
                asls_lam=float(asls_lam),
                asls_p=float(asls_p),
            )
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.plot(two_theta, intensity, lw=0.8, alpha=0.45, label="原始谱线")
            ax.plot(two_theta, y_proc, lw=1.3, label="处理后谱线")
            ax.axvspan(range_002[0], range_002[1], alpha=0.08, color="tab:red", label="(002)拟合区间")
            ax.axvspan(range_100[0], range_100[1], alpha=0.08, color="tab:green", label="(100)拟合区间")
            ax.axvline(result.peak_002.two_theta, color="tab:red", ls="--", label="(002)中心")
            ax.axvline(result.peak_100.two_theta, color="tab:green", ls="--", label="(100)中心")
            ax.set_xlabel("2θ (degree)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.legend(ncol=2)
            ax.grid(alpha=0.2)
            st.pyplot(fig)

            results_rows.append(
                {
                    "file": up.name,
                    "d002_nm": result.d002_nm,
                    "Lc_nm": result.lc_nm,
                    "La_nm": result.la_nm,
                    "fit_range_002": f"{range_002[0]:.2f}-{range_002[1]:.2f}",
                    "fit_range_100": f"{range_100[0]:.2f}-{range_100[1]:.2f}",
                    "smooth_mode": ("none" if not smooth else smooth_mode),
                    "baseline_mode": baseline_mode,
                    "peak002_2theta_deg": result.peak_002.two_theta,
                    "peak100_2theta_deg": result.peak_100.two_theta,
                }
            )

    if results_rows:
        st.subheader("📋 批量结果汇总")
        summary_df = pd.DataFrame(results_rows)
        st.dataframe(summary_df, use_container_width=True)
        st.download_button("下载全部结果CSV", summary_df.to_csv(index=False).encode("utf-8"), "xrd_batch_result.csv", "text/csv")
else:
    st.info("请先上传数据文件。")
