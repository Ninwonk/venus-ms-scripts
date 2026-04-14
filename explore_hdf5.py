"""
HDF5 文件结构探测脚本
====================
遍历 .strc (HDF5) 文件的内部结构，打印所有 groups、datasets、shapes、dtypes，
用于验证数据格式是否符合预期（14列二维数组，前两列为 XY 坐标）。
"""

import sys
import h5py
import numpy as np


def explore_hdf5(filepath: str) -> None:
    """递归遍历 HDF5 文件，打印完整的层级结构信息。"""
    with h5py.File(filepath, "r") as f:
        print(f"{'='*60}")
        print(f"文件: {filepath}")
        print(f"{'='*60}")

        def _visit(name, obj):
            """h5py visititems 回调：打印每个节点的详细信息。"""
            indent = "  " * name.count("/")
            if isinstance(obj, h5py.Group):
                print(f"{indent}📁 Group: /{name}  (成员数: {len(obj)})")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}📊 Dataset: /{name}")
                print(f"{indent}   Shape: {obj.shape}")
                print(f"{indent}   Dtype: {obj.dtype}")
                print(f"{indent}   Size:  {obj.size} 个元素")

                # 如果是二维数组，打印前几行作为预览
                if obj.ndim == 2:
                    preview_rows = min(3, obj.shape[0])
                    print(f"{indent}   列数:  {obj.shape[1]}")
                    print(f"{indent}   前 {preview_rows} 行预览:")
                    data = obj[:preview_rows, :]
                    for i, row in enumerate(data):
                        # 格式化显示，保留4位小数
                        row_str = ", ".join(f"{v:.4f}" for v in row)
                        print(f"{indent}     [{i}] [{row_str}]")

                    # 打印每列的统计信息
                    print(f"\n{indent}   各列统计信息:")
                    all_data = obj[:]
                    for col_idx in range(obj.shape[1]):
                        col = all_data[:, col_idx]
                        print(
                            f"{indent}     列{col_idx:2d}: "
                            f"min={col.min():.4f}, "
                            f"max={col.max():.4f}, "
                            f"mean={col.mean():.4f}, "
                            f"std={col.std():.4f}"
                        )

                elif obj.ndim == 1:
                    preview = min(10, obj.shape[0])
                    print(f"{indent}   前 {preview} 个元素: {obj[:preview]}")

            # 打印属性（如果有）
            if hasattr(obj, "attrs") and len(obj.attrs) > 0:
                indent_attr = "  " * (name.count("/") + 1)
                print(f"{indent_attr}属性:")
                for key, val in obj.attrs.items():
                    print(f"{indent_attr}  {key} = {val}")

        # 打印根级属性
        if len(f.attrs) > 0:
            print("根属性:")
            for key, val in f.attrs.items():
                print(f"  {key} = {val}")
            print()

        # 递归遍历所有节点
        f.visititems(_visit)

        print(f"\n{'='*60}")
        print("探测完毕")
        print(f"{'='*60}")


if __name__ == "__main__":
    # 默认探测当前目录下的 .strc 文件
    default_file = r"20251023_080353.strc"
    filepath = sys.argv[1] if len(sys.argv) > 1 else default_file
    explore_hdf5(filepath)
