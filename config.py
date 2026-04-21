"""
配置模块
========
集中管理所有可配置参数，包括：
- HDF5 文件路径和 dataset 路径
- 传感器配对关系（入射光/反射光通道）
- 每对传感器的光效率系数
- 衍射效率合格范围
- 网格插值参数
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class SpotConfig:
    """
    单个激光光斑的配置。

    Attributes:
        name: 光斑名称（用于显示）
        incident_channel: 入射光传感器通道编号（1-based，对应数据文件第3~14列）
        reflected_channel: 反射光传感器通道编号（1-based）
        incident_efficiency: 入射光效率系数（默认 1.0）
        reflected_efficiency: 反射光效率系数（默认 1.0）
    """
    name: str
    incident_channel: int       # 1-based 传感器编号
    reflected_channel: int      # 1-based 传感器编号
    incident_efficiency: float = 1.0
    reflected_efficiency: float = 1.0


@dataclass
class AppConfig:
    """
    应用全局配置。

    Attributes:
        hdf5_dataset_path: HDF5 文件内 dataset 的路径
        spots: 光斑配置列表（每个光斑包含入射/反射通道及效率系数）
        pass_range: 衍射效率合格范围 (min, max)，用于热力图颜色标识
        grid_resolution: 网格插值的分辨率（X 和 Y 方向的网格点数），
                         None 表示自动使用数据本身的唯一坐标数
        colormap: 热力图使用的 colormap 名称
    """
    hdf5_dataset_path: str = "/scandata"

    # 默认光斑配对关系：(入射光通道, 反射光通道)
    # 通道编号为 1-based，对应传感器列（数据文件第3~14列分别为通道1~12）
    spots: List[SpotConfig] = field(default_factory=lambda: [
        SpotConfig(
            name="光斑1",
            incident_channel=5,
            reflected_channel=9,
            reflected_efficiency=2.2748,
        ),
        SpotConfig(
            name="光斑2",
            incident_channel=6,
            reflected_channel=10,
            reflected_efficiency=2.15598,
        ),
        SpotConfig(
            name="光斑3",
            incident_channel=7,
            reflected_channel=11,
            reflected_efficiency=2.18705,
        ),
        SpotConfig(
            name="光斑4",
            incident_channel=8,
            reflected_channel=12,
            reflected_efficiency=2.11911,
        ),
        SpotConfig(name="光斑5", incident_channel=1, reflected_channel=2),
        SpotConfig(name="光斑6", incident_channel=3, reflected_channel=4),
    ])

    # 衍射效率合格范围 (最小值, 最大值)
    pass_range: Tuple[float, float] = (0.5, 1.0)

    # 网格插值分辨率，None 表示使用数据原始的唯一坐标点
    grid_resolution: Optional[Tuple[int, int]] = None

    # matplotlib 热力图 colormap
    colormap: str = "RdYlGn"

    def get_spot_by_name(self, name: str) -> Optional[SpotConfig]:
        """根据名称查找光斑配置。"""
        for spot in self.spots:
            if spot.name == name:
                return spot
        return None

    def get_spot_names(self) -> List[str]:
        """返回所有光斑的名称列表。"""
        return [s.name for s in self.spots]


# 全局默认配置实例
DEFAULT_CONFIG = AppConfig()
