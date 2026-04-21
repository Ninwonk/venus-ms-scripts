"""
核心分析模块
============
提供衍射效率计算、网格化、差分计算等核心数据处理功能。

主要功能：
1. compute_diffraction_efficiency — 根据入射/反射通道数据计算衍射效率
2. build_grid — 将散点数据（不等间距 XY）组织为规则二维网格
3. compute_differential — 两个光斑的衍射效率经位置偏移对齐后做差分
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull, QhullError
from typing import Tuple, Optional, Dict, Any

from config import SpotConfig, AppConfig, DEFAULT_CONFIG
from data_loader import get_channel_column


def compute_diffraction_efficiency(
    df: pd.DataFrame,
    spot: SpotConfig,
) -> np.ndarray:
    """
    计算指定光斑的衍射效率。

    公式: DE = (反射光值 × 反射光效率系数) / (入射光值 × 入射光效率系数)

    Parameters:
        df: 包含传感器数据的 DataFrame
        spot: 光斑配置（包含入射/反射通道编号和效率系数）

    Returns:
        衍射效率数组，与 df 行数相同
    """
    # 获取入射光和反射光列名
    inc_col = get_channel_column(spot.incident_channel)
    ref_col = get_channel_column(spot.reflected_channel)

    # 提取通道数据
    incident = df[inc_col].values
    reflected = df[ref_col].values

    # 计算衍射效率，避免除零（入射光为 0 时结果设为 NaN）
    numerator = reflected * spot.reflected_efficiency
    denominator = incident * spot.incident_efficiency

    with np.errstate(divide="ignore", invalid="ignore"):
        de = np.where(denominator != 0, numerator / denominator, np.nan)

    return de


def build_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    resolution: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将散点数据组织为规则二维网格。

    利用数据本身的 XY 坐标结构（通常已是规则间距扫描数据），
    通过 pivot 操作构建网格。如果数据不完整，缺失位置填充 NaN。

    Parameters:
        x: X 坐标数组 (N,)
        y: Y 坐标数组 (N,)
        values: 对应的数值数组 (N,)
        resolution: 可选的网格分辨率 (nx, ny)。
                    如果指定，会对数据进行重采样到该分辨率；
                    如果为 None，则使用数据本身的唯一坐标值。

    Returns:
        (x_unique, y_unique, grid_2d):
            x_unique: 排序后的唯一 X 坐标 (nx,)
            y_unique: 排序后的唯一 Y 坐标 (ny,)
            grid_2d: 二维网格数据 (ny, nx)，行对应 Y，列对应 X
    """
    # 将 XY 坐标四舍五入以消除浮点精度问题（扫描步进精度为 0.01）
    x_rounded = np.round(x, decimals=4)
    y_rounded = np.round(y, decimals=4)

    # 构建临时 DataFrame 用于 pivot
    temp_df = pd.DataFrame({"x": x_rounded, "y": y_rounded, "value": values})

    # 如果同一 (x, y) 位置有多个测量值，取平均
    temp_df = temp_df.groupby(["x", "y"], as_index=False)["value"].mean()

    # Pivot 为二维网格：行=Y, 列=X
    pivot = temp_df.pivot(index="y", columns="x", values="value")
    pivot = pivot.sort_index(ascending=True)  # Y 从小到大排序
    pivot = pivot[sorted(pivot.columns)]       # X 从小到大排序

    x_unique = np.array(pivot.columns, dtype=float)
    y_unique = np.array(pivot.index, dtype=float)
    grid_2d = pivot.values  # shape: (ny, nx)

    # 如果指定了分辨率，重采样到目标网格
    if resolution is not None:
        nx, ny = resolution
        grid_2d, x_unique, y_unique = _resample_grid(
            x_unique, y_unique, grid_2d, nx, ny
        )

    return x_unique, y_unique, grid_2d


def _resample_grid(
    x_orig: np.ndarray,
    y_orig: np.ndarray,
    grid_orig: np.ndarray,
    nx: int,
    ny: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将已有的规则网格重采样到指定分辨率。

    使用 RegularGridInterpolator 进行线性插值。

    Parameters:
        x_orig, y_orig: 原始网格的坐标轴
        grid_orig: 原始二维网格数据 (ny_orig, nx_orig)
        nx, ny: 目标分辨率

    Returns:
        (x_new, y_new, grid_new)
    """
    # 用 NaN 填充缺失值会导致插值失败，先用最近邻填充
    grid_filled = _fill_nan_nearest(grid_orig)

    # 创建插值器（y 对应 axis 0, x 对应 axis 1）
    interpolator = RegularGridInterpolator(
        (y_orig, x_orig), grid_filled,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    # 生成新的等间距网格
    x_new = np.linspace(x_orig.min(), x_orig.max(), nx)
    y_new = np.linspace(y_orig.min(), y_orig.max(), ny)
    yy, xx = np.meshgrid(y_new, x_new, indexing="ij")
    points = np.column_stack([yy.ravel(), xx.ravel()])

    grid_new = interpolator(points).reshape(ny, nx)
    return x_new, y_new, grid_new


def _fill_nan_nearest(grid: np.ndarray) -> np.ndarray:
    """
    用最近邻方法填充网格中的 NaN 值。

    Parameters:
        grid: 可能包含 NaN 的二维数组

    Returns:
        填充后的数组副本
    """
    from scipy.ndimage import distance_transform_edt

    result = grid.copy()
    mask = np.isnan(result)

    if not mask.any():
        return result

    # 计算每个 NaN 位置到最近有效值的索引
    _, nearest_idx = distance_transform_edt(mask, return_distances=True, return_indices=True)
    result[mask] = result[tuple(nearest_idx[:, mask])]
    return result


def compute_differential(
    df: pd.DataFrame,
    spot_a: SpotConfig,
    spot_b: SpotConfig,
    dx: float,
    dy: float,
    resolution: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算两个光斑衍射效率的差分值。

    步骤：
    1. 分别计算 spot_a 和 spot_b 的衍射效率
    2. 将各自的衍射效率构建为二维网格
    3. 将 spot_b 的网格坐标偏移 (dx, dy)
    4. 将两个网格插值到统一的公共网格上
    5. 对应网格点相减：diff = DE_a - DE_b

    Parameters:
        df: 包含传感器数据的 DataFrame
        spot_a: 第一个光斑配置
        spot_b: 第二个光斑配置
        dx: X 方向位置偏移（spot_b 相对于 spot_a 的偏移量）
        dy: Y 方向位置偏移（spot_b 相对于 spot_a 的偏移量）
        resolution: 公共网格的分辨率 (nx, ny)，None 时自动确定

    Returns:
        (x_common, y_common, diff_grid):
            x_common: 公共网格 X 坐标
            y_common: 公共网格 Y 坐标
            diff_grid: 差分结果二维网格 (ny, nx)
    """
    x = df["x"].values
    y = df["y"].values

    # 1. 分别计算衍射效率
    de_a = compute_diffraction_efficiency(df, spot_a)
    de_b = compute_diffraction_efficiency(df, spot_b)

    # 2. 构建各自的二维网格
    x_a, y_a, grid_a = build_grid(x, y, de_a)
    x_b, y_b, grid_b = build_grid(x, y, de_b)

    # 3. 确定偏移后 spot_b 的坐标范围
    x_b_shifted = x_b + dx
    y_b_shifted = y_b + dy

    # 4. 确定公共重叠区域
    x_min = max(x_a.min(), x_b_shifted.min())
    x_max = min(x_a.max(), x_b_shifted.max())
    y_min = max(y_a.min(), y_b_shifted.min())
    y_max = min(y_a.max(), y_b_shifted.max())

    if x_min >= x_max or y_min >= y_max:
        raise ValueError(
            f"偏移 (dx={dx}, dy={dy}) 后两个网格没有重叠区域，无法计算差分"
        )

    # 5. 生成公共网格
    if resolution is not None:
        nx, ny = resolution
    else:
        # 使用 spot_a 在重叠区域内的坐标点数作为分辨率参考
        nx = int(np.sum((x_a >= x_min) & (x_a <= x_max)))
        ny = int(np.sum((y_a >= y_min) & (y_a <= y_max)))
        # 确保至少有 2 个点
        nx = max(nx, 2)
        ny = max(ny, 2)

    x_common = np.linspace(x_min, x_max, nx)
    y_common = np.linspace(y_min, y_max, ny)

    # 6. 用 NaN 填充后创建插值器
    grid_a_filled = _fill_nan_nearest(grid_a)
    grid_b_filled = _fill_nan_nearest(grid_b)

    interp_a = RegularGridInterpolator(
        (y_a, x_a), grid_a_filled,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    # 注意：spot_b 的插值器使用偏移后的坐标
    interp_b = RegularGridInterpolator(
        (y_b_shifted, x_b_shifted), grid_b_filled,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    # 7. 在公共网格上求值并相减
    yy, xx = np.meshgrid(y_common, x_common, indexing="ij")
    points = np.column_stack([yy.ravel(), xx.ravel()])

    values_a = interp_a(points).reshape(ny, nx)
    values_b = interp_b(points).reshape(ny, nx)

    diff_grid = values_a - values_b

    return x_common, y_common, diff_grid


@dataclass
class CalibrationResult:
    """单个光斑光栅区域拟合结果。"""
    center_x: float
    center_y: float
    width: float
    height: float
    area: float
    angle_deg: float
    mask_area: float
    fill_ratio: float
    is_partial: bool
    size_error: Optional[float]
    boundary_point_count: int
    corners: np.ndarray
    selected_mask: np.ndarray
    threshold_mask: np.ndarray
    smoothed_grid: np.ndarray


def _estimate_step(values: np.ndarray) -> float:
    """估计扫描轴步进（使用去重后差分的中位数）。"""
    unique_vals = np.sort(np.unique(values))
    if unique_vals.size < 2:
        return 1.0
    diffs = np.diff(unique_vals)
    diffs = diffs[np.abs(diffs) > 1e-9]
    if diffs.size == 0:
        return 1.0
    return float(np.median(np.abs(diffs)))


def _fit_rotated_rectangle(points_xy: np.ndarray) -> Dict[str, Any]:
    """
    对二维点集拟合旋转矩形。

    优先使用“凸包 + 最小面积包围矩形”拟合；
    当点集退化（共线/数量不足）时回退到 PCA 方案。

    返回矩形中心、宽高、角度与四角点（按顺时针或逆时针）。
    """
    if points_xy.shape[0] < 3:
        raise ValueError("可用于拟合的点数量不足（<3）")

    unique_points = np.unique(points_xy, axis=0)
    if unique_points.shape[0] < 3:
        raise ValueError("可用于拟合的唯一点数量不足（<3）")

    def _fit_by_pca(pts: np.ndarray) -> Dict[str, Any]:
        center = pts.mean(axis=0)
        centered = pts - center

        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        axis_u = eigvecs[:, order[0]]
        axis_v = eigvecs[:, order[1]]

        if axis_u[0] < 0:
            axis_u = -axis_u
        if np.linalg.det(np.column_stack([axis_u, axis_v])) < 0:
            axis_v = -axis_v

        proj_u = centered @ axis_u
        proj_v = centered @ axis_v
        min_u, max_u = float(np.min(proj_u)), float(np.max(proj_u))
        min_v, max_v = float(np.min(proj_v)), float(np.max(proj_v))
        width = max_u - min_u
        height = max_v - min_v

        center_u = (min_u + max_u) / 2.0
        center_v = (min_v + max_v) / 2.0
        rect_center = center + center_u * axis_u + center_v * axis_v

        corners_uv = np.array(
            [[min_u, min_v], [max_u, min_v], [max_u, max_v], [min_u, max_v]],
            dtype=float,
        )
        corners_xy = rect_center + (
            (corners_uv - np.array([center_u, center_v])) @ np.column_stack([axis_u, axis_v]).T
        )
        angle_deg = float(np.degrees(np.arctan2(axis_u[1], axis_u[0])))
        while angle_deg >= 90.0:
            angle_deg -= 180.0
        while angle_deg < -90.0:
            angle_deg += 180.0

        return {
            "center_x": float(rect_center[0]),
            "center_y": float(rect_center[1]),
            "width": float(width),
            "height": float(height),
            "area": float(width * height),
            "angle_deg": angle_deg,
            "corners": corners_xy,
        }

    try:
        hull = ConvexHull(unique_points)
        hull_pts = unique_points[hull.vertices]
    except QhullError:
        return _fit_by_pca(unique_points)

    if hull_pts.shape[0] < 3:
        return _fit_by_pca(unique_points)

    best: Optional[Dict[str, Any]] = None
    best_area = np.inf

    n = hull_pts.shape[0]
    for i in range(n):
        p0 = hull_pts[i]
        p1 = hull_pts[(i + 1) % n]
        edge = p1 - p0
        edge_norm = float(np.hypot(edge[0], edge[1]))
        if edge_norm < 1e-12:
            continue

        ux, uy = edge[0] / edge_norm, edge[1] / edge_norm
        vx, vy = -uy, ux
        axis_u = np.array([ux, uy], dtype=float)
        axis_v = np.array([vx, vy], dtype=float)

        proj_u = hull_pts @ axis_u
        proj_v = hull_pts @ axis_v

        min_u, max_u = float(np.min(proj_u)), float(np.max(proj_u))
        min_v, max_v = float(np.min(proj_v)), float(np.max(proj_v))
        width = max_u - min_u
        height = max_v - min_v
        area = width * height
        if area >= best_area:
            continue

        center_u = (min_u + max_u) / 2.0
        center_v = (min_v + max_v) / 2.0
        rect_center = center_u * axis_u + center_v * axis_v

        corners_uv = np.array(
            [[min_u, min_v], [max_u, min_v], [max_u, max_v], [min_u, max_v]],
            dtype=float,
        )
        corners_xy = corners_uv @ np.column_stack([axis_u, axis_v]).T

        angle_deg = float(np.degrees(np.arctan2(axis_u[1], axis_u[0])))
        if height > width:
            width, height = height, width
            angle_deg += 90.0
        while angle_deg >= 90.0:
            angle_deg -= 180.0
        while angle_deg < -90.0:
            angle_deg += 180.0

        best_area = area
        best = {
            "center_x": float(rect_center[0]),
            "center_y": float(rect_center[1]),
            "width": float(width),
            "height": float(height),
            "area": float(width * height),
            "angle_deg": angle_deg,
            "corners": corners_xy,
        }

    if best is None:
        return _fit_by_pca(unique_points)
    return best


def _choose_transition_threshold(
    outside_value: float,
    threshold_min: float,
    threshold_max: float,
) -> Optional[float]:
    """
    根据边界外侧点的值判断应使用哪条阈值边界做插值。

    - outside < threshold_min: 使用下阈值边界
    - outside > threshold_max: 使用上阈值边界
    - outside 在阈值带内: 不属于真实阈值边界（常见于 ROI 截断），返回 None
    """
    if outside_value < threshold_min:
        return threshold_min
    if outside_value > threshold_max:
        return threshold_max
    return None


def _collect_subgrid_boundary_points(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    value_grid: np.ndarray,
    component_mask: np.ndarray,
    valid_mask: np.ndarray,
    threshold_min: float,
    threshold_max: float,
) -> np.ndarray:
    """
    从连通域边界提取亚网格插值点。

    仅在“连通域内点 vs 连通域外点”的网格边上采样：
    - 当外侧值越过阈值边界时，沿该边做线性插值，得到阈值交点（亚网格精度）
    - ROI 截断导致的伪边界（外侧仍在阈值带内）会被跳过
    """
    ny, nx = component_mask.shape
    points = []

    def _append_interp_point(
        in_r: int,
        in_c: int,
        out_r: int,
        out_c: int,
    ) -> None:
        if not (valid_mask[in_r, in_c] and valid_mask[out_r, out_c]):
            return

        inside_value = float(value_grid[in_r, in_c])
        outside_value = float(value_grid[out_r, out_c])
        threshold = _choose_transition_threshold(
            outside_value,
            threshold_min,
            threshold_max,
        )
        if threshold is None:
            return

        denom = outside_value - inside_value
        if abs(denom) < 1e-12:
            t = 0.5
        else:
            t = (threshold - inside_value) / denom
        if not np.isfinite(t):
            return
        t = float(np.clip(t, 0.0, 1.0))

        x0, y0 = float(x_grid[in_c]), float(y_grid[in_r])
        x1, y1 = float(x_grid[out_c]), float(y_grid[out_r])
        px = x0 + t * (x1 - x0)
        py = y0 + t * (y1 - y0)
        points.append((px, py))

    # 水平边（左-右）
    for r in range(ny):
        for c in range(nx - 1):
            left_in = bool(component_mask[r, c])
            right_in = bool(component_mask[r, c + 1])
            if left_in == right_in:
                continue
            if left_in:
                _append_interp_point(r, c, r, c + 1)
            else:
                _append_interp_point(r, c + 1, r, c)

    # 垂直边（上-下）
    for r in range(ny - 1):
        for c in range(nx):
            top_in = bool(component_mask[r, c])
            bottom_in = bool(component_mask[r + 1, c])
            if top_in == bottom_in:
                continue
            if top_in:
                _append_interp_point(r, c, r + 1, c)
            else:
                _append_interp_point(r + 1, c, r, c)

    if not points:
        return np.empty((0, 2), dtype=float)

    pts = np.asarray(points, dtype=float)
    # 去重（边界角点可能重复）
    rounded = np.round(pts, decimals=8)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    return pts[np.sort(unique_idx)]


def calibrate_grating_rectangle(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    de_grid: np.ndarray,
    threshold_min: float,
    threshold_max: float,
    median_kernel_size: int = 3,
    min_component_points: int = 20,
    roi: Optional[Tuple[float, float, float, float]] = None,
    expected_size: Optional[Tuple[float, float]] = None,
) -> CalibrationResult:
    """
    在单个光斑的衍射效率网格中提取光栅区域并拟合旋转矩形。

    Parameters:
        x_grid, y_grid: 网格坐标轴（来自原始扫描 XY）
        de_grid: 衍射效率二维网格 (ny, nx)
        threshold_min/max: 光栅区域阈值
        median_kernel_size: 中值滤波核大小（建议奇数；1 表示不滤波）
        min_component_points: 最小连通域点数
        roi: 可选 ROI (x_min, x_max, y_min, y_max)
        expected_size: 可选预期尺寸 (width, height)，仅用于评分与误差输出

    Note:
        拟合优先使用“阈值边界上的亚网格插值点”；当边界点过少时回退为网格点拟合。
    """
    if threshold_min > threshold_max:
        raise ValueError("阈值下限不能大于上限")

    if median_kernel_size < 1:
        raise ValueError("median_kernel_size 必须 >= 1")

    if median_kernel_size % 2 == 0:
        median_kernel_size += 1

    valid_mask = np.isfinite(de_grid)
    if not np.any(valid_mask):
        raise ValueError("网格数据全为 NaN，无法标定")

    grid_filled = _fill_nan_nearest(de_grid)
    if median_kernel_size > 1:
        smoothed = ndimage.median_filter(grid_filled, size=median_kernel_size, mode="nearest")
    else:
        smoothed = grid_filled.copy()

    threshold_mask = (
        valid_mask
        & (smoothed >= threshold_min)
        & (smoothed <= threshold_max)
    )

    row_min_limit, row_max_limit = 0, de_grid.shape[0] - 1
    col_min_limit, col_max_limit = 0, de_grid.shape[1] - 1

    if roi is not None:
        x_min, x_max, y_min, y_max = roi
        if x_min > x_max or y_min > y_max:
            raise ValueError("ROI 范围非法：最小值不能大于最大值")
        yy, xx = np.meshgrid(y_grid, x_grid, indexing="ij")
        roi_mask = (
            (xx >= x_min) & (xx <= x_max)
            & (yy >= y_min) & (yy <= y_max)
        )
        roi_rows, roi_cols = np.where(roi_mask)
        if roi_rows.size == 0 or roi_cols.size == 0:
            raise ValueError("ROI 区域不在当前扫描网格内")
        row_min_limit, row_max_limit = int(roi_rows.min()), int(roi_rows.max())
        col_min_limit, col_max_limit = int(roi_cols.min()), int(roi_cols.max())
        threshold_mask = threshold_mask & roi_mask

    if not np.any(threshold_mask):
        raise ValueError("阈值分割后未找到候选区域，请调整阈值或 ROI")

    labeled, num_labels = ndimage.label(threshold_mask)
    if num_labels == 0:
        raise ValueError("未检测到连通区域")

    dx = _estimate_step(x_grid)
    dy = _estimate_step(y_grid)
    point_area = dx * dy

    expected_sorted = None
    if expected_size is not None:
        ew, eh = expected_size
        if ew <= 0 or eh <= 0:
            raise ValueError("expected_size 必须为正值")
        expected_sorted = sorted([float(ew), float(eh)])

    best = None
    best_score = -np.inf

    for label_id in range(1, num_labels + 1):
        component = labeled == label_id
        idx = np.argwhere(component)
        n_points = idx.shape[0]
        if n_points < min_component_points:
            continue

        rows = idx[:, 0]
        cols = idx[:, 1]
        grid_points_xy = np.column_stack([x_grid[cols], y_grid[rows]])

        boundary_points = _collect_subgrid_boundary_points(
            x_grid=x_grid,
            y_grid=y_grid,
            value_grid=smoothed,
            component_mask=component,
            valid_mask=valid_mask,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
        if boundary_points.shape[0] >= 4:
            fit_points_xy = boundary_points
        else:
            fit_points_xy = grid_points_xy

        rect = _fit_rotated_rectangle(fit_points_xy)
        rect_area = rect["area"]
        if rect_area <= 0:
            continue

        mask_area = float(n_points * point_area)
        fill_ratio = float(mask_area / rect_area) if rect_area > 0 else np.nan

        touches_border = bool(
            (rows.min() == row_min_limit)
            or (rows.max() == row_max_limit)
            or (cols.min() == col_min_limit)
            or (cols.max() == col_max_limit)
        )

        size_error = None
        size_weight = 1.0
        if expected_sorted is not None:
            fitted_sorted = sorted([rect["width"], rect["height"]])
            size_error = (
                abs(fitted_sorted[0] - expected_sorted[0]) / expected_sorted[0]
                + abs(fitted_sorted[1] - expected_sorted[1]) / expected_sorted[1]
            ) / 2.0
            size_weight = 1.0 / (1.0 + size_error)

        # 评分：优先面积较大、填充率较高；若提供 expected_size 则加权
        score = mask_area * max(fill_ratio, 1e-6) * size_weight

        if score > best_score:
            best_score = score
            best = {
                "rect": rect,
                "mask_area": mask_area,
                "fill_ratio": fill_ratio,
                "is_partial": touches_border,
                "size_error": size_error,
                "component": component,
                "boundary_point_count": int(boundary_points.shape[0]),
            }

    if best is None:
        raise ValueError(
            "候选连通区域都过小，无法拟合。请降低最小连通域点数或调整阈值。"
        )

    rect = best["rect"]
    return CalibrationResult(
        center_x=rect["center_x"],
        center_y=rect["center_y"],
        width=rect["width"],
        height=rect["height"],
        area=rect["area"],
        angle_deg=rect["angle_deg"],
        mask_area=best["mask_area"],
        fill_ratio=best["fill_ratio"],
        is_partial=best["is_partial"],
        size_error=best["size_error"],
        boundary_point_count=best["boundary_point_count"],
        corners=rect["corners"],
        selected_mask=best["component"],
        threshold_mask=threshold_mask,
        smoothed_grid=smoothed,
    )
