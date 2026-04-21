"""
Streamlit Web 应用
==================
基于 Streamlit 的交互式传感器数据分析应用。
提供数据概览、衍射效率热力图、差分热力图三个功能页。

启动方式:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

from config import AppConfig, SpotConfig
from data_loader import load_strc_file, get_file_metadata
from analysis import (
    compute_diffraction_efficiency,
    build_grid,
    compute_differential,
    calibrate_grating_rectangle,
)
from visualization import (
    plot_heatmap_plotly,
    plot_diff_heatmap_plotly,
    plot_calibration_result_plotly,
)


# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="传感器扫描数据分析",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 传感器扫描数据分析")
st.caption("HDF5 (.strc) 传感器数据 → 衍射效率热力图 & 差分分析")


# ============================================================
# 侧边栏：文件选择 & 传感器配对配置
# ============================================================
with st.sidebar:
    st.header("📂 数据文件")

    # 文件选择：支持上传或使用默认文件
    upload_mode = st.radio(
        "数据来源",
        ["使用本地文件", "上传文件"],
        index=0,
    )

    data_file_path = None
    if upload_mode == "使用本地文件":
        # 扫描当前目录下的 .strc 文件
        strc_files = [f for f in os.listdir(".") if f.endswith(".strc")]
        if strc_files:
            data_file_path = st.selectbox("选择文件", strc_files)
        else:
            st.warning("当前目录下没有 .strc 文件")
    else:
        uploaded = st.file_uploader("上传 .strc 文件", type=["strc", "hdf5", "h5"])
        if uploaded is not None:
            # 保存上传的文件到临时路径（分块写入，避免大文件撑爆内存）
            temp_path = f"_uploaded_{uploaded.name}"
            if not os.path.exists(temp_path):
                with open(temp_path, "wb") as f:
                    while chunk := uploaded.read(8 * 1024 * 1024):  # 8MB 分块
                        f.write(chunk)
            data_file_path = temp_path

    st.divider()

    # ---- 传感器配对配置 ----
    st.header("⚙️ 传感器配对")

    # 光斑数量
    num_spots = st.number_input("光斑数量", min_value=1, max_value=6, value=6)

    # 默认配对关系
    default_pairs = [
        (5, 9), (6, 10), (7, 11), (8, 12), (1, 2), (3, 4),
    ]
    default_ref_eff = [2.2748, 2.15598, 2.18705, 2.11911, 1.0, 1.0]

    spots = []
    for i in range(num_spots):
        with st.expander(f"光斑{i+1} 配置", expanded=(i == 0)):
            default_inc = default_pairs[i][0] if i < len(default_pairs) else 1
            default_ref = default_pairs[i][1] if i < len(default_pairs) else 2
            default_ref_eff_i = (
                default_ref_eff[i] if i < len(default_ref_eff) else 1.0
            )

            col1, col2 = st.columns(2)
            with col1:
                inc_ch = st.number_input(
                    f"入射光通道", min_value=1, max_value=12,
                    value=default_inc, key=f"inc_{i}",
                )
            with col2:
                ref_ch = st.number_input(
                    f"反射光通道", min_value=1, max_value=12,
                    value=default_ref, key=f"ref_{i}",
                )

            col3, col4 = st.columns(2)
            with col3:
                inc_eff = st.number_input(
                    "入射光效率", min_value=0.001, value=1.0,
                    format="%.4f", key=f"inc_eff_{i}",
                )
            with col4:
                ref_eff = st.number_input(
                    "反射光效率", min_value=0.001, value=default_ref_eff_i,
                    format="%.5f", key=f"ref_eff_{i}",
                )

            spots.append(SpotConfig(
                name=f"光斑{i+1}",
                incident_channel=inc_ch,
                reflected_channel=ref_ch,
                incident_efficiency=inc_eff,
                reflected_efficiency=ref_eff,
            ))

    # 构建配置对象
    app_config = AppConfig(spots=spots)


# ============================================================
# 数据加载（使用 st.cache_data 缓存避免重复读取）
# ============================================================

@st.cache_data(show_spinner="正在加载数据...")
def cached_load_data(filepath: str) -> pd.DataFrame:
    """缓存加载 .strc 文件数据。"""
    return load_strc_file(filepath, AppConfig())


@st.cache_data(show_spinner=False)
def cached_get_metadata(filepath: str) -> dict:
    """缓存获取文件元数据。"""
    return get_file_metadata(filepath, AppConfig())


# ============================================================
# 主界面：三个 Tab
# ============================================================
if data_file_path is None:
    st.info("👈 请在侧边栏选择或上传数据文件")
    st.stop()

# 加载数据
df = cached_load_data(data_file_path)
metadata = cached_get_metadata(data_file_path)
spot_names = app_config.get_spot_names()

# 创建四个标签页
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 数据概览",
    "🌡️ 衍射效率热力图",
    "📊 差分热力图",
    "🎯 离线光斑标定",
])


# ---- Tab 1: 数据概览 ----
with tab1:
    st.subheader("文件元数据")
    meta_cols = st.columns(4)
    for i, (k, v) in enumerate(metadata.items()):
        meta_cols[i % 4].metric(k, str(v))

    st.subheader("数据统计")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("总行数", f"{len(df):,}")
        st.metric("X 范围", f"[{df['x'].min():.2f}, {df['x'].max():.2f}]")
    with col_b:
        st.metric("总列数", str(len(df.columns)))
        st.metric("Y 范围", f"[{df['y'].min():.2f}, {df['y'].max():.2f}]")

    st.subheader("数据预览 (前 100 行)")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("各通道统计")
    stats = df.describe().T
    st.dataframe(stats, use_container_width=True)


# ---- Tab 2: 衍射效率热力图 ----
with tab2:
    st.subheader("衍射效率热力图")

    # 控制栏
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

    with ctrl_col1:
        selected_spot_name = st.selectbox("选择光斑", spot_names)

    with ctrl_col2:
        pass_min = st.number_input(
            "合格范围下限", value=0.5, format="%.4f", key="pass_min",
        )

    with ctrl_col3:
        pass_max = st.number_input(
            "合格范围上限", value=1.0, format="%.4f", key="pass_max",
        )

    with ctrl_col4:
        use_pass_range = st.checkbox("启用合格范围标识", value=True)

    # 获取选中的光斑配置
    selected_spot = app_config.get_spot_by_name(selected_spot_name)

    if st.button("🔥 生成热力图", type="primary", key="btn_heatmap"):
        with st.spinner("正在计算衍射效率并生成热力图..."):
            # 计算衍射效率
            de = compute_diffraction_efficiency(df, selected_spot)

            # 构建网格
            x_grid, y_grid, grid_2d = build_grid(
                df["x"].values, df["y"].values, de,
            )

            # 统计信息
            valid_de = de[~np.isnan(de)]
            stat_cols = st.columns(4)
            stat_cols[0].metric("最小值", f"{valid_de.min():.4f}")
            stat_cols[1].metric("最大值", f"{valid_de.max():.4f}")
            stat_cols[2].metric("平均值", f"{valid_de.mean():.4f}")
            stat_cols[3].metric("标准差", f"{valid_de.std():.4f}")

            # 合格范围
            pass_range = (pass_min, pass_max) if use_pass_range else None

            # 绘制
            fig = plot_heatmap_plotly(
                x_grid, y_grid, grid_2d,
                title=f"{selected_spot.name} 衍射效率 "
                      f"(入射ch{selected_spot.incident_channel} / "
                      f"反射ch{selected_spot.reflected_channel})",
                pass_range=pass_range,
            )
            st.plotly_chart(fig, use_container_width=True)

            # 合格率统计
            if use_pass_range:
                in_range = np.sum(
                    (valid_de >= pass_min) & (valid_de <= pass_max)
                )
                total = len(valid_de)
                pass_rate = in_range / total * 100 if total > 0 else 0
                st.info(
                    f"合格率: {pass_rate:.2f}% "
                    f"({in_range:,} / {total:,} 个数据点在 "
                    f"[{pass_min:.4f}, {pass_max:.4f}] 范围内)"
                )


# ---- Tab 3: 差分热力图 ----
with tab3:
    st.subheader("差分热力图")
    st.caption("差分 = 光斑A 衍射效率 - 光斑B 衍射效率（偏移对齐后）")

    # 控制栏
    diff_col1, diff_col2, diff_col3, diff_col4 = st.columns(4)

    with diff_col1:
        spot_a_name = st.selectbox("光斑 A", spot_names, index=0, key="diff_a")
    with diff_col2:
        spot_b_name = st.selectbox(
            "光斑 B", spot_names,
            index=min(1, len(spot_names) - 1),
            key="diff_b",
        )
    with diff_col3:
        diff_dx = st.number_input("X 偏移 dx (mm)", value=0.0, format="%.4f")
    with diff_col4:
        diff_dy = st.number_input("Y 偏移 dy (mm)", value=0.0, format="%.4f")

    if st.button("📊 生成差分热力图", type="primary", key="btn_diff"):
        spot_a = app_config.get_spot_by_name(spot_a_name)
        spot_b = app_config.get_spot_by_name(spot_b_name)

        with st.spinner("正在计算差分（大数据量可能需要较长时间）..."):
            try:
                x_diff, y_diff, diff_grid = compute_differential(
                    df, spot_a, spot_b, dx=diff_dx, dy=diff_dy,
                )

                # 统计信息
                valid_diff = diff_grid[~np.isnan(diff_grid)]
                stat_cols = st.columns(4)
                stat_cols[0].metric("最小差分", f"{valid_diff.min():.4f}")
                stat_cols[1].metric("最大差分", f"{valid_diff.max():.4f}")
                stat_cols[2].metric("平均差分", f"{valid_diff.mean():.4f}")
                stat_cols[3].metric("标准差", f"{valid_diff.std():.4f}")

                # 绘制
                fig = plot_diff_heatmap_plotly(
                    x_diff, y_diff, diff_grid,
                    title=f"差分: {spot_a.name} - {spot_b.name} "
                          f"(偏移 dx={diff_dx}, dy={diff_dy})",
                )
                st.plotly_chart(fig, use_container_width=True)

            except ValueError as e:
                st.error(f"计算错误: {e}")


# ---- Tab 4: 离线光斑标定 ----
with tab4:
    st.subheader("离线光斑位置标定")
    st.caption(
        "流程：阈值分割 → 连通域筛选 → 旋转矩形拟合 → 输出矩形面积与中心坐标"
    )

    calib_col1, calib_col2, calib_col3, calib_col4 = st.columns(4)
    with calib_col1:
        selected_spots = st.multiselect(
            "参与标定的光斑",
            options=spot_names,
            default=spot_names[:min(2, len(spot_names))],
            key="calib_spots",
        )
    with calib_col2:
        threshold_min = st.number_input(
            "光栅阈值下限",
            value=0.5000,
            format="%.4f",
            key="calib_th_min",
        )
    with calib_col3:
        threshold_max = st.number_input(
            "光栅阈值上限",
            value=1.0000,
            format="%.4f",
            key="calib_th_max",
        )
    with calib_col4:
        median_kernel = st.selectbox(
            "去噪核大小",
            options=[1, 3, 5, 7],
            index=1,
            key="calib_median_kernel",
            help="中值滤波核大小；1 表示不滤波。",
        )

    calib_col5, calib_col6, calib_col7 = st.columns([1, 1, 2])
    with calib_col5:
        min_component_points = st.number_input(
            "最小连通域点数",
            min_value=1,
            max_value=1000000,
            value=30,
            step=1,
            key="calib_min_points",
        )
    with calib_col6:
        use_expected_size = st.checkbox(
            "启用尺寸对比", value=False, key="calib_use_expected"
        )
    with calib_col7:
        use_roi = st.checkbox(
            "启用手动 ROI（数值框选）", value=False, key="calib_use_roi"
        )

    expected_size = None
    if use_expected_size:
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            expected_width = st.number_input(
                "预期光栅宽度 (mm)",
                min_value=0.0001,
                value=1.0000,
                format="%.4f",
                key="calib_exp_width",
            )
        with exp_col2:
            expected_height = st.number_input(
                "预期光栅高度 (mm)",
                min_value=0.0001,
                value=1.0000,
                format="%.4f",
                key="calib_exp_height",
            )
        expected_size = (expected_width, expected_height)

    roi = None
    if use_roi:
        x_min_data, x_max_data = float(df["x"].min()), float(df["x"].max())
        y_min_data, y_max_data = float(df["y"].min()), float(df["y"].max())

        roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
        with roi_col1:
            roi_x_min = st.number_input(
                "ROI X最小",
                value=x_min_data,
                format="%.4f",
                key="calib_roi_x_min",
            )
        with roi_col2:
            roi_x_max = st.number_input(
                "ROI X最大",
                value=x_max_data,
                format="%.4f",
                key="calib_roi_x_max",
            )
        with roi_col3:
            roi_y_min = st.number_input(
                "ROI Y最小",
                value=y_min_data,
                format="%.4f",
                key="calib_roi_y_min",
            )
        with roi_col4:
            roi_y_max = st.number_input(
                "ROI Y最大",
                value=y_max_data,
                format="%.4f",
                key="calib_roi_y_max",
            )
        roi = (
            min(roi_x_min, roi_x_max),
            max(roi_x_min, roi_x_max),
            min(roi_y_min, roi_y_max),
            max(roi_y_min, roi_y_max),
        )

    if st.button("🎯 执行离线标定", type="primary", key="btn_calibration"):
        if not selected_spots:
            st.warning("请至少选择一个光斑")
        elif threshold_min > threshold_max:
            st.error("阈值下限不能大于上限")
        else:
            with st.spinner("正在进行离线标定..."):
                x_values = df["x"].values
                y_values = df["y"].values
                summary_rows = []

                for spot_name in selected_spots:
                    spot_cfg = app_config.get_spot_by_name(spot_name)
                    if spot_cfg is None:
                        summary_rows.append({
                            "光斑": spot_name,
                            "状态": "配置缺失",
                        })
                        continue

                    de = compute_diffraction_efficiency(df, spot_cfg)
                    x_grid, y_grid, de_grid = build_grid(x_values, y_values, de)

                    try:
                        result = calibrate_grating_rectangle(
                            x_grid=x_grid,
                            y_grid=y_grid,
                            de_grid=de_grid,
                            threshold_min=threshold_min,
                            threshold_max=threshold_max,
                            median_kernel_size=median_kernel,
                            min_component_points=min_component_points,
                            roi=roi,
                            expected_size=expected_size,
                        )
                    except ValueError as e:
                        st.error(f"{spot_name} 标定失败：{e}")
                        summary_rows.append({
                            "光斑": spot_name,
                            "状态": f"失败: {e}",
                        })
                        continue

                    fig = plot_calibration_result_plotly(
                        x=x_grid,
                        y=y_grid,
                        grid=result.smoothed_grid,
                        corners=result.corners,
                        center=(result.center_x, result.center_y),
                        threshold_mask=result.threshold_mask,
                        selected_mask=result.selected_mask,
                        threshold_range=(threshold_min, threshold_max),
                        title=f"{spot_name} 光栅拟合结果",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    status = "部分覆盖（触边）" if result.is_partial else "正常"
                    row = {
                        "光斑": spot_name,
                        "状态": status,
                        "中心X(mm)": result.center_x,
                        "中心Y(mm)": result.center_y,
                        "矩形宽(mm)": result.width,
                        "矩形高(mm)": result.height,
                        "矩形面积(mm²)": result.area,
                        "区域面积(mm²)": result.mask_area,
                        "填充率": result.fill_ratio,
                        "角度(°)": result.angle_deg,
                        "边界亚网格点数": result.boundary_point_count,
                    }
                    if result.size_error is not None:
                        row["尺寸误差"] = result.size_error

                    summary_rows.append(row)

            if summary_rows:
                st.subheader("标定结果汇总")
                result_df = pd.DataFrame(summary_rows)
                st.dataframe(result_df, use_container_width=True)
