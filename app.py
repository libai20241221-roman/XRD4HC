import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from xrd_analysis import analyze_hard_carbon_xrd, preprocess_intensity, suggest_peak_ranges
from xrd_io import read_xrd_table_from_upload

st.set_page_config(page_title="硬炭XRD快速分析", layout="wide")
st.title("🧪 硬炭 XRD 数据分析（La / Lc / d002）")
st.caption("支持批量上传；每个文件可独立覆盖参数、独立分析范围与显示范围。")


def _peak_attr(peak, name: str, default: float = float("nan")) -> float:
    v = getattr(peak, name, default)
    try:
        return float(v)
    except Exception:
        return default


def grade_quality(r2_002: float, r2_100: float, rmse_002: float, rmse_100: float) -> str:
    avg_r2 = (r2_002 + r2_100) / 2
    avg_rmse = (rmse_002 + rmse_100) / 2
    if avg_r2 >= 0.995 and avg_rmse <= 0.03 * 1000:
        return "A"
    if avg_r2 >= 0.985:
        return "B"
    if avg_r2 >= 0.97:
        return "C"
    return "D"


with st.sidebar:
    st.header("全局默认参数")
    mode = st.radio("分析模式", ["一键快速自动分析", "手动高级分析"], index=0)
    lambda_nm = st.number_input("X射线波长 λ (nm)", value=0.15406, format="%.5f")
    k_lc = st.number_input("Scherrer常数 Kc (Lc)", value=0.89, format="%.3f")
    k_la = st.number_input("Scherrer常数 Ka (La)", value=1.84, format="%.3f")
    inst_fwhm = st.number_input("仪器展宽FWHM (°2θ)", value=0.0, format="%.4f")

    smooth = st.checkbox("启用平滑", value=True)
    smooth_mode = st.selectbox("平滑模式", ["savgol", "moving_average", "none"])
    sg_window = st.number_input("S-G窗口", min_value=5, max_value=101, value=11, step=2)
    sg_poly = st.number_input("S-G阶数", min_value=1, max_value=7, value=3)
    ma_window = st.number_input("移动平均窗口", min_value=3, max_value=101, value=7)

    baseline_mode = st.selectbox("背景模式", ["percentile", "rolling_min", "poly", "asls", "none"])
    poly_deg = st.number_input("多项式阶数", min_value=1, max_value=5, value=2)
    asls_lam = st.number_input("AsLS λ", min_value=1e2, max_value=1e8, value=1e5, format="%.0f")
    asls_p = st.number_input("AsLS p", min_value=0.0001, max_value=0.5, value=0.01, format="%.4f")

    with st.expander("📘 背景处理原理与教程", expanded=False):
        st.markdown("""
- percentile: constant low-percentile baseline for flat background.  
- rolling_min: moving minimum baseline for slowly varying background.  
- poly: polynomial trend baseline for curved background.  
- asls: Asymmetric Least Squares, robust for complex broad background.  

Recommended flow: `percentile` → `rolling_min/poly` → `asls` if needed.
""")

uploaded_files = st.file_uploader("上传文件（可多选）", type=["csv", "txt", "dat", "xlsx", "xls"], accept_multiple_files=True)

if uploaded_files:
    rows = []
    tabs = st.tabs([f"文件{i+1}: {f.name}" for i, f in enumerate(uploaded_files)])

    for tab, up in zip(tabs, uploaded_files):
        with tab:
            try:
                two_theta, intensity = read_xrd_table_from_upload(up.name, up.read())
            except Exception as e:
                st.error(f"解析失败: {e}")
                continue

            st.markdown("### 单文件独立调整")
            use_override = st.checkbox("启用该文件独立参数", key=f"ov_{up.name}")

            x_min_default, x_max_default = float(two_theta.min()), float(two_theta.max())
            cxa, cxb = st.columns(2)
            with cxa:
                x_left = st.number_input(f"X范围左边界 (°2θ) [{up.name}]", value=x_min_default, format="%.2f", key=f"xleft_{up.name}")
            with cxb:
                x_right = st.number_input(f"X范围右边界 (°2θ) [{up.name}]", value=x_max_default, format="%.2f", key=f"xright_{up.name}")
            x_range = (float(min(x_left, x_right)), float(max(x_left, x_right)))

            tt = two_theta
            yy = intensity
            m_ana = (tt >= x_range[0]) & (tt <= x_range[1])
            tt_ana, yy_ana = tt[m_ana], yy[m_ana]
            if len(tt_ana) < 20:
                st.warning("分析范围内数据点太少，请扩大范围。")
                continue

            lmbd, klc, kla, ifwhm = lambda_nm, k_lc, k_la, inst_fwhm
            sm, sm_mode, bmode = smooth, smooth_mode, baseline_mode
            sgw, sgp, maw = int(sg_window), int(sg_poly), int(ma_window)
            pdg, alam, ap = int(poly_deg), float(asls_lam), float(asls_p)

            if use_override:
                c1, c2 = st.columns(2)
                with c1:
                    lmbd = st.number_input(f"λ [{up.name}]", value=lmbd, format="%.5f", key=f"l_{up.name}")
                    klc = st.number_input(f"Kc [{up.name}]", value=klc, format="%.3f", key=f"kc_{up.name}")
                    kla = st.number_input(f"Ka [{up.name}]", value=kla, format="%.3f", key=f"ka_{up.name}")
                    ifwhm = st.number_input(f"Inst FWHM [{up.name}]", value=ifwhm, format="%.4f", key=f"if_{up.name}")
                with c2:
                    sm = st.checkbox(f"平滑 [{up.name}]", value=sm, key=f"sm_{up.name}")
                    sm_mode = st.selectbox(f"平滑模式 [{up.name}]", ["savgol", "moving_average", "none"], index=["savgol","moving_average","none"].index(sm_mode), key=f"smm_{up.name}")
                    bmode = st.selectbox(f"背景模式 [{up.name}]", ["percentile", "rolling_min", "poly", "asls", "none"], index=["percentile","rolling_min","poly","asls","none"].index(bmode), key=f"bm_{up.name}")

            if mode == "一键快速自动分析":
                range_002, range_100 = suggest_peak_ranges(tt_ana, preprocess_intensity(yy_ana, smooth=sm, window=sgw, polyorder=sgp, baseline_mode=bmode, smooth_mode=("none" if not sm else sm_mode), ma_window=maw, baseline_poly_degree=pdg, asls_lam=alam, asls_p=ap))
            else:
                s002 = st.slider(f"(002) 拟合区间 [{up.name}]", min_value=float(tt_ana.min()), max_value=float(tt_ana.max()), value=(max(float(tt_ana.min()),20.0), min(float(tt_ana.max()),32.0)), step=0.1, key=f"r2_{up.name}")
                s100 = st.slider(f"(100) 拟合区间 [{up.name}]", min_value=float(tt_ana.min()), max_value=float(tt_ana.max()), value=(max(float(tt_ana.min()),38.0), min(float(tt_ana.max()),52.0)), step=0.1, key=f"r1_{up.name}")
                range_002, range_100 = (float(s002[0]), float(s002[1])), (float(s100[0]), float(s100[1]))

            try:
                res = analyze_hard_carbon_xrd(tt_ana, yy_ana, lambda_nm=lmbd, k_lc=klc, k_la=kla, inst_fwhm_deg=ifwhm, smooth=sm, range_002=range_002, range_100=range_100, baseline_mode=bmode, smooth_mode=("none" if not sm else sm_mode), ma_window=maw, baseline_poly_degree=pdg, asls_lam=alam, asls_p=ap)
            except Exception as e:
                st.error(f"拟合失败: {e}")
                continue

            p002_r2 = _peak_attr(res.peak_002, "r2")
            p100_r2 = _peak_attr(res.peak_100, "r2")
            p002_rmse = _peak_attr(res.peak_002, "rmse")
            p100_rmse = _peak_attr(res.peak_100, "rmse")
            quality = grade_quality(p002_r2, p100_r2, p002_rmse, p100_rmse)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("d002 (nm)", f"{res.d002_nm:.4f}")
            c2.metric("Lc (nm)", f"{res.lc_nm:.3f}")
            c3.metric("La (nm)", f"{res.la_nm:.3f}")
            c4.metric("Fit Grade", quality)

            st.dataframe(pd.DataFrame({
                "Peak": ["(002)", "(100)"],
                "Center 2θ (°)": [res.peak_002.two_theta, res.peak_100.two_theta],
                "FWHM (°2θ)": [res.peak_002.fwhm_deg, res.peak_100.fwhm_deg],
                "R²": [p002_r2, p100_r2],
                "RMSE": [p002_rmse, p100_rmse],
            }), use_container_width=True)

            y_proc = preprocess_intensity(yy_ana, smooth=sm, window=sgw, polyorder=sgp, baseline_mode=bmode, smooth_mode=("none" if not sm else sm_mode), ma_window=maw, baseline_poly_degree=pdg, asls_lam=alam, asls_p=ap)
            m_disp = (tt_ana >= x_range[0]) & (tt_ana <= x_range[1])
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.plot(tt_ana[m_disp], yy_ana[m_disp], lw=0.8, alpha=0.45, label="Raw")
            ax.plot(tt_ana[m_disp], y_proc[m_disp], lw=1.3, label="Processed")
            ax.axvline(res.peak_002.two_theta, color="tab:red", ls="--", label="(002) center")
            ax.axvline(res.peak_100.two_theta, color="tab:green", ls="--", label="(100) center")
            ax.set_xlim(x_range)
            ax.legend(ncol=2)
            ax.grid(alpha=0.2)
            st.pyplot(fig)

            rows.append({
                "File": up.name,
                "d002 (nm)": res.d002_nm,
                "Lc (nm)": res.lc_nm,
                "La (nm)": res.la_nm,
                "Fit Grade": quality,
                "R² (002)": p002_r2,
                "R² (100)": p100_r2,
                "X range (°2θ)": f"{x_range[0]:.1f}-{x_range[1]:.1f}",
            })

    if rows:
        st.subheader("📋 批量结果汇总")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button("下载全部结果CSV", df.to_csv(index=False).encode("utf-8"), "xrd_batch_result.csv", "text/csv")
else:
    st.info("请先上传数据文件。")
