"""
数据加载模块
============
负责读取 .strc (HDF5) 格式的传感器扫描数据文件，
将原始数据转换为结构化的 pandas DataFrame。

数据文件格式：
- HDF5 文件，dataset 路径由 config 指定（默认 /scandata）
- 14 列 float32 二维数组
- 列 0: X 坐标
- 列 1: Y 坐标
- 列 2~13: 传感器通道 1~12 的测量值
"""

import h5py
import numpy as np
import pandas as pd

from config import AppConfig, DEFAULT_CONFIG


# DataFrame 中的列名映射
COLUMN_NAMES = ["x", "y"] + [f"ch{i}" for i in range(1, 13)]


def load_strc_file(
    filepath: str,
    config: AppConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """
    读取 .strc (HDF5) 文件并返回 DataFrame。

    Parameters:
        filepath: .strc 文件的路径
        config: 应用配置，用于获取 dataset 路径

    Returns:
        pandas DataFrame，列名为 ['x', 'y', 'ch1', 'ch2', ..., 'ch12']
    """
    with h5py.File(filepath, "r") as f:
        dataset = f[config.hdf5_dataset_path]
        # 一次性读取所有数据到内存（float32 → float64 由 pandas 自动处理）
        raw_data = dataset[:]

    # 验证列数是否符合预期
    expected_cols = 14  # 2 (XY) + 12 (传感器)
    if raw_data.shape[1] != expected_cols:
        raise ValueError(
            f"数据列数不匹配：预期 {expected_cols} 列，实际 {raw_data.shape[1]} 列"
        )

    # 构建 DataFrame
    df = pd.DataFrame(raw_data, columns=COLUMN_NAMES)
    return df


def get_file_metadata(filepath: str, config: AppConfig = DEFAULT_CONFIG) -> dict:
    """
    读取 HDF5 文件的元数据属性。

    Parameters:
        filepath: .strc 文件路径
        config: 应用配置

    Returns:
        包含元数据的字典，如 XStart, XStep, XEnd, YStart, YStep, YEnd, SensorCount
    """
    import json

    with h5py.File(filepath, "r") as f:
        dataset = f[config.hdf5_dataset_path]
        metadata = {}

        # 读取 dataset 上的属性
        for key in dataset.attrs:
            val = dataset.attrs[key]
            # 如果值是 bytes 类型，尝试解析为 JSON
            if isinstance(val, bytes):
                try:
                    metadata.update(json.loads(val.decode("utf-8")))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    metadata[key] = val.decode("utf-8", errors="replace")
            else:
                metadata[key] = val

    return metadata


def get_channel_column(channel_number: int) -> str:
    """
    将 1-based 传感器通道编号转换为 DataFrame 列名。

    Parameters:
        channel_number: 传感器通道编号 (1~12)

    Returns:
        对应的列名，如 'ch1', 'ch2', ...

    Raises:
        ValueError: 通道编号超出范围
    """
    if not 1 <= channel_number <= 12:
        raise ValueError(f"通道编号必须在 1~12 之间，实际: {channel_number}")
    return f"ch{channel_number}"
