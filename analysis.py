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
from scipy.interpolate import RegularGridInterpolator
from typing import Tuple, Optional

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
