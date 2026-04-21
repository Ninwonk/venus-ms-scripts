"""
可视化模块
==========
提供 matplotlib（用于 Notebook）和 plotly（用于 Web）两套热力图绘制函数。

功能：
1. 衍射效率热力图 — 用颜色区间标识合格/不合格范围
2. 差分热力图 — 展示两个光斑衍射效率的差异
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from typing import Tuple, Optional

import plotly.graph_objects as go


# ============================================================
# 颜色映射工具函数
# ============================================================

def _build_pass_fail_colormap(
    pass_range: Tuple[float, float],
    vmin: float,
    vmax: float,
    base_cmap: str = "RdYlGn",
) -> Tuple[mcolors.Colormap, mcolors.Normalize]:
    """
    构建带合格/不合格区间标识的自定义 colormap。

    合格范围内使用绿色系渐变，不合格范围使用红色系。
    通过在 colormap 上设置分段颜色来实现。

    Parameters:
        pass_range: 合格范围 (min_pass, max_pass)
        vmin: 数据显示的最小值
        vmax: 数据显示的最大值
        base_cmap: 基础 colormap 名称

    Returns:
        (colormap, norm): matplotlib 的 colormap 和 Normalize 对象
    """
    pass_min, pass_max = pass_range

    # 使用 BoundaryNorm 创建分段颜色映射
    # 定义边界：[vmin, pass_min, pass_max, vmax]
    boundaries = sorted(set([vmin, pass_min, pass_max, vmax]))

    # 为每个区间定义颜色
    colors = []
    for i in range(len(boundaries) - 1):
        low, high = boundaries[i], boundaries[i + 1]
        mid = (low + high) / 2
        if mid < pass_min:
            colors.append("#d73027")  # 红色 - 低于合格范围
        elif mid > pass_max:
            colors.append("#4575b4")  # 蓝色 - 高于合格范围
        else:
            colors.append("#1a9850")  # 绿色 - 合格范围内

    if len(colors) == 0:
        colors = ["#1a9850"]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    return cmap, norm


def _build_plotly_colorscale(
    pass_range: Tuple[float, float],
    vmin: float,
    vmax: float,
) -> list:
    """
    构建 Plotly 热力图的 colorscale，标识合格/不合格范围。

    Parameters:
        pass_range: 合格范围 (min_pass, max_pass)
        vmin: 数据最小值
        vmax: 数据最大值

    Returns:
        Plotly 格式的 colorscale 列表
    """
    pass_min, pass_max = pass_range
    data_range = vmax - vmin

    if data_range <= 0:
        return [[0, "#1a9850"], [1, "#1a9850"]]

    # 将合格范围映射到 [0, 1] 区间，并严格限制在 [0, 1]
    norm_pass_min = (pass_min - vmin) / data_range
    norm_pass_max = (pass_max - vmin) / data_range

    # 数据全部高于合格范围上限 → 全蓝
    if norm_pass_max <= 0:
        return [[0, "#4575b4"], [1, "#4575b4"]]

    # 数据全部低于合格范围下限 → 全红
    if norm_pass_min >= 1:
        return [[0, "#d73027"], [1, "#d73027"]]

    # 裁剪到 [0, 1]
    norm_pass_min = max(0.0, norm_pass_min)
    norm_pass_max = min(1.0, norm_pass_max)

    colorscale = []

    # 低于合格范围的区域 — 红色
    if norm_pass_min > 0:
        colorscale.append([0, "#d73027"])
        colorscale.append([norm_pass_min, "#d73027"])

    # 合格范围内 — 绿色渐变
    colorscale.append([norm_pass_min, "#1a9850"])
    colorscale.append([norm_pass_max, "#66bd63"])

    # 高于合格范围的区域 — 蓝色
    if norm_pass_max < 1:
        colorscale.append([norm_pass_max, "#4575b4"])
        colorscale.append([1, "#4575b4"])

    return colorscale


# ============================================================
# Matplotlib 绘图函数（用于 Jupyter Notebook）
# ============================================================

def plot_heatmap_mpl(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    title: str = "衍射效率热力图",
    pass_range: Optional[Tuple[float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "RdYlGn",
    figsize: Tuple[int, int] = (12, 8),
) -> Figure:
    """
    使用 matplotlib 绘制热力图。

    Parameters:
        x: X 轴坐标数组 (nx,)
        y: Y 轴坐标数组 (ny,)
        grid: 二维数据网格 (ny, nx)
        title: 图表标题
        pass_range: 合格范围 (min, max)，None 表示不区分合格/不合格
        vmin: colorbar 最小值，None 则自动
        vmax: colorbar 最大值，None 则自动
        cmap: 默认 colormap 名称（当 pass_range 为 None 时使用）
        figsize: 图表大小 (宽, 高) 英寸

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 确定数据范围
    data_vmin = np.nanmin(grid) if vmin is None else vmin
    data_vmax = np.nanmax(grid) if vmax is None else vmax

    # 根据是否有合格范围选择 colormap
    if pass_range is not None:
        colormap, norm = _build_pass_fail_colormap(
            pass_range, data_vmin, data_vmax
        )
        im = ax.pcolormesh(
            x, y, grid,
            cmap=colormap, norm=norm,
            shading="nearest",
        )
    else:
        im = ax.pcolormesh(
            x, y, grid,
            cmap=cmap,
            vmin=data_vmin, vmax=data_vmax,
            shading="nearest",
        )

    # 添加 colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("衍射效率 (DE)")

    # 如果有合格范围，在图上标注
    if pass_range is not None:
        ax.set_title(
            f"{title}\n合格范围: [{pass_range[0]:.4f}, {pass_range[1]:.4f}]",
            fontsize=14,
        )
        # 在 colorbar 上标注合格范围线
        cbar.ax.axhline(y=pass_range[0], color="white", linewidth=1.5, linestyle="--")
        cbar.ax.axhline(y=pass_range[1], color="white", linewidth=1.5, linestyle="--")
    else:
        ax.set_title(title, fontsize=14)

    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


def plot_diff_heatmap_mpl(
    x: np.ndarray,
    y: np.ndarray,
    diff_grid: np.ndarray,
    title: str = "差分热力图",
    figsize: Tuple[int, int] = (12, 8),
    symmetric_range: bool = True,
) -> Figure:
    """
    使用 matplotlib 绘制差分热力图。

    差分图通常使用发散型 colormap（蓝-白-红），
    0 值居中，正负值对称显示。

    Parameters:
        x: X 轴坐标数组
        y: Y 轴坐标数组
        diff_grid: 差分二维数据网格
        title: 图表标题
        figsize: 图表大小
        symmetric_range: 是否对称显示范围（以 0 为中心）

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 确定数据范围
    if symmetric_range:
        abs_max = np.nanmax(np.abs(diff_grid))
        vmin, vmax = -abs_max, abs_max
    else:
        vmin = np.nanmin(diff_grid)
        vmax = np.nanmax(diff_grid)

    im = ax.pcolormesh(
        x, y, diff_grid,
        cmap="RdBu_r",  # 发散型 colormap: 红(正) - 白(零) - 蓝(负)
        vmin=vmin, vmax=vmax,
        shading="nearest",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("差分值 (ΔDE)")

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X (mm)", fontsize=12)
    ax.set_ylabel("Y (mm)", fontsize=12)
    ax.set_aspect("equal")

    plt.tight_layout()
    return fig


# ============================================================
# Plotly 绘图函数（用于 Streamlit Web 应用）
# ============================================================

def plot_heatmap_plotly(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    title: str = "衍射效率热力图",
    pass_range: Optional[Tuple[float, float]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> go.Figure:
    """
    使用 Plotly 绘制交互式热力图。

    Parameters:
        x: X 轴坐标数组
        y: Y 轴坐标数组
        grid: 二维数据网格 (ny, nx)
        title: 图表标题
        pass_range: 合格范围 (min, max)，None 不区分
        vmin: 最小值
        vmax: 最大值

    Returns:
        plotly Figure 对象
    """
    data_vmin = float(np.nanmin(grid)) if vmin is None else vmin
    data_vmax = float(np.nanmax(grid)) if vmax is None else vmax

    # 构建 colorscale
    if pass_range is not None:
        colorscale = _build_plotly_colorscale(pass_range, data_vmin, data_vmax)
    else:
        colorscale = "RdYlGn"

    fig = go.Figure(data=go.Heatmap(
        x=x,
        y=y,
        z=grid,
        colorscale=colorscale,
        zmin=data_vmin,
        zmax=data_vmax,
        colorbar=dict(title="DE"),
        hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>DE: %{z:.4f}<extra></extra>",
    ))

    # 标题中附带合格范围信息
    title_text = title
    if pass_range is not None:
        title_text += f"<br><sub>合格范围: [{pass_range[0]:.4f}, {pass_range[1]:.4f}]</sub>"

    fig.update_layout(
        title=title_text,
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        yaxis_scaleanchor="x",  # 保持 X/Y 等比例
        width=900,
        height=700,
    )

    return fig


def plot_diff_heatmap_plotly(
    x: np.ndarray,
    y: np.ndarray,
    diff_grid: np.ndarray,
    title: str = "差分热力图",
    symmetric_range: bool = True,
) -> go.Figure:
    """
    使用 Plotly 绘制交互式差分热力图。

    Parameters:
        x: X 轴坐标数组
        y: Y 轴坐标数组
        diff_grid: 差分二维数据网格
        title: 图表标题
        symmetric_range: 是否对称范围

    Returns:
        plotly Figure 对象
    """
    if symmetric_range:
        abs_max = float(np.nanmax(np.abs(diff_grid)))
        zmin, zmax = -abs_max, abs_max
    else:
        zmin = float(np.nanmin(diff_grid))
        zmax = float(np.nanmax(diff_grid))

    fig = go.Figure(data=go.Heatmap(
        x=x,
        y=y,
        z=diff_grid,
        colorscale="RdBu_r",  # 发散型配色
        zmid=0,
        zmin=zmin,
        zmax=zmax,
        colorbar=dict(title="ΔDE"),
        hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>ΔDE: %{z:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        yaxis_scaleanchor="x",
        width=900,
        height=700,
    )

    return fig


def plot_calibration_result_plotly(
    x: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
    corners: np.ndarray,
    center: Tuple[float, float],
    title: str = "光栅区域拟合结果",
    threshold_mask: Optional[np.ndarray] = None,
    selected_mask: Optional[np.ndarray] = None,
    threshold_range: Optional[Tuple[float, float]] = None,
) -> go.Figure:
    """
    绘制离线光斑标定结果：
    - DE 热力图
    - 阈值分割轮廓（可选）
    - 最终选中连通域轮廓（可选）
    - 旋转矩形与中心点
    """
    data_vmin = float(np.nanmin(grid))
    data_vmax = float(np.nanmax(grid))

    if threshold_range is not None:
        colorscale = _build_plotly_colorscale(threshold_range, data_vmin, data_vmax)
    else:
        colorscale = "RdYlGn"

    fig = go.Figure(data=go.Heatmap(
        x=x,
        y=y,
        z=grid,
        colorscale=colorscale,
        zmin=data_vmin,
        zmax=data_vmax,
        colorbar=dict(title="DE"),
        hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>DE: %{z:.4f}<extra></extra>",
    ))

    if threshold_mask is not None and np.any(threshold_mask):
        fig.add_trace(go.Contour(
            x=x,
            y=y,
            z=threshold_mask.astype(float),
            contours=dict(
                start=0.5,
                end=0.5,
                size=1.0,
                coloring="none",
            ),
            line=dict(color="rgba(255,255,255,0.6)", width=1.5),
            showscale=False,
            hoverinfo="skip",
            name="阈值区域",
        ))

    if selected_mask is not None and np.any(selected_mask):
        fig.add_trace(go.Contour(
            x=x,
            y=y,
            z=selected_mask.astype(float),
            contours=dict(
                start=0.5,
                end=0.5,
                size=1.0,
                coloring="none",
            ),
            line=dict(color="#00e5ff", width=2.5),
            showscale=False,
            hoverinfo="skip",
            name="拟合区域",
        ))

    corners_closed = np.vstack([corners, corners[0]])
    fig.add_trace(go.Scatter(
        x=corners_closed[:, 0],
        y=corners_closed[:, 1],
        mode="lines",
        line=dict(color="#ffbf00", width=3),
        name="拟合矩形",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=[center[0]],
        y=[center[1]],
        mode="markers",
        marker=dict(size=10, color="#ff0055", symbol="x"),
        name="矩形中心",
        hovertemplate="中心 X: %{x:.4f}<br>中心 Y: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        yaxis_scaleanchor="x",
        width=900,
        height=700,
        legend=dict(orientation="h", y=1.02, x=0.0),
    )

    return fig
