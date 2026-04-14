# Venus MS Scripts

传感器扫描数据分析工具 — 解析 `.strc` (HDF5) 格式的传感器扫描数据，计算衍射效率并生成热力图。

## 功能

- **衍射效率计算** — 根据可配置的入射/反射光通道和效率系数，计算衍射效率 `DE = (反射光 × 反射光效率) / (入射光 × 入射光效率)`
- **热力图生成** — 基于 XY 坐标网格绘制衍射效率热力图，支持合格/不合格颜色区间标识
- **差分分析** — 两个光斑衍射效率经位置偏移对齐后做差分，绘制差分热力图
- **双版本** — 提供 Jupyter Notebook 和 Streamlit Web 两种使用方式

## 项目结构

```
├── config.py                 # 可配置参数（传感器配对、效率系数、合格范围）
├── data_loader.py            # HDF5 (.strc) 数据加载
├── analysis.py               # 核心计算（衍射效率、网格化、差分）
├── visualization.py          # 绘图模块（matplotlib + plotly）
├── analysis_notebook.ipynb   # Jupyter Notebook 版本
├── app.py                    # Streamlit Web 应用
├── explore_hdf5.py           # HDF5 文件结构探测工具
├── smoke_test.py             # 端到端冒烟测试
└── requirements.txt          # Python 依赖
```

## 快速开始

### 1. 创建虚拟环境并安装依赖

```bash
uv venv --python 3.13
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

uv pip install -r requirements.txt
```

### 2. 使用 Jupyter Notebook

在 VSCode 中打开 `analysis_notebook.ipynb`，选择 `.venv` 内核，按 Cell 顺序运行：

1. **配置参数** — 设置文件路径、传感器配对关系、效率系数
2. **加载数据** — 读取 HDF5 文件并预览
3. **衍射效率热力图** — 选择光斑，计算并绘制
4. **差分热力图** — 选择两个光斑，设置偏移量

### 3. 使用 Streamlit Web 应用

```bash
streamlit run app.py
```

在浏览器中打开后，通过侧边栏配置传感器参数，三个 Tab 分别提供：
- 📋 数据概览
- 🌡️ 衍射效率热力图
- 📊 差分热力图

## 配置说明

### 传感器配对（默认）

| 光斑 | 入射光通道 | 反射光通道 |
|------|-----------|-----------|
| 1    | 1         | 2         |
| 2    | 3         | 4         |
| 3    | 5         | 9         |
| 4    | 6         | 10        |
| 5    | 7         | 11        |
| 6    | 8         | 12        |

配对关系在 Notebook 中通过 `SpotConfig` 配置，在 Web 中通过侧边栏 UI 动态配置。

### 可调参数

- **入射光/反射光效率系数** — 每对传感器独立设置，默认 1.0
- **合格范围** — 衍射效率的 min/max 阈值，热力图用颜色区间标识
- **位置偏移 (dx, dy)** — 差分计算时两个光斑的物理距离差

## 数据文件格式

`.strc` 文件为 HDF5 格式（改了扩展名），内部结构：
- Dataset 路径: `/scandata`
- 14 列 float32 二维数组: 列 0-1 为 XY 坐标，列 2-13 为 12 个传感器通道值

可使用 `explore_hdf5.py` 探测文件结构：

```bash
python explore_hdf5.py your_file.strc
```
