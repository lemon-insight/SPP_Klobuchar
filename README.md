# GPS 单点定位算法实现

本项目实现了基于伪距的GPS单点定位算法，包含带电离层改正（Klobuchar模型）和不带电离层改正两种版本，并提供RINEX观测文件解析功能。

## 文件结构

```
d:\卫星导航算法大作业\SPP？Klobuchar\
├── code/
│   ├── parse_rinex_obs.py          # RINEX观测文件解析器
│   ├── single_point_positioning_no_ionosphere.py  # 无电离层改正的SPP
│   ├── single_point_positioning_klobuchar_numpy.py # Klobuchar电离层改正的SPP
│   ├── batch_process.py            # 批处理脚本
│   ├── batch_positioning.py        # 批量定位脚本
│   └── batch_positioning_afternoon.py  # 中午时段批量定位脚本
├── DATA/
│   ├── DATA_original/
│   │   ├── nav/                    # 原始星历文件
│   │   └── obs/                    # 原始RINEX观测文件
│   ├── DATA_readed/
│   │   ├── obs_data/               # 解析后的观测数据（CSV格式）
│   │   └── ephemeris/              # 星历和电离层参数文件
│   └── DATA_zip/                   # 压缩的CRX文件
├── output/
│   ├── RESULT_morning/             # 上午时段定位结果
│   └── RESULT_afternoon/           # 中午时段定位结果
├── text/                           # 测试结果文件
├── RNXCMP/                         # RINEX格式转换工具
└── README.md
```

## 文件说明

### 1. parse_rinex_obs.py

**功能**：解析RINEX 3.x格式的GNSS观测文件，提取GPS卫星的C1C伪距观测值。

**主要类**：

- `RinexObsParser`: RINEX文件解析器
  - `parse()`: 解析整个RINEX文件
  - `_parse_header()`: 解析文件头信息（版本、测站名、近似坐标、观测类型）
  - `_parse_observations()`: 解析观测数据历元
  - `_compute_gps_week()`: 计算GPS周和周内秒

**输出格式**：

- `*_report.txt`: 可读的解析报告
- `*_meta.txt`: 元数据文件（含测站近似坐标）
- `*_obs.csv`: 观测数据CSV文件（便于单点定位计算）

### 2. single_point_positioning_no_ionosphere.py

**功能**：不带电离层改正的GPS单点定位实现。

**主要类**：

- `SPPSolver`: 单点定位求解器
  - `__init__(obs_file, ephemeris_file, approx_coords, debug=False, time_system='GPS')`
  - `process_file(max_epochs=2880, debug=False)`: 处理整个观测文件
  - `solve_epoch(epoch_data, initial_coords=None, debug=False)`: 解算单个历元

**输入参数**：

- `obs_file`: 观测数据CSV文件路径
- `ephemeris_file`: 卫星星历CSV文件路径
- `approx_coords`: 接收机近似坐标 [X, Y, Z]（ITRF坐标系，单位：米）
- `debug`: 是否输出调试信息（默认False）
- `time_system`: 时间系统（'GPS' 或 'UTC'，默认'GPS'）

**输出结果**（每个历元）：

- X, Y, Z: 解算坐标（米）
- dX, dY, dZ: 相对于近似坐标的偏差（米）
- Clock (us): 接收机钟差（微秒）
- Satellites: 使用的卫星数量
- RMS (m): 残差均方根
- Max Res (m): 最大残差
- Status: 解算状态（Success/Failure）

### 3. single_point_positioning_klobuchar_numpy.py

**功能**：带Klobuchar电离层改正的GPS单点定位实现。

**主要类**：

- `SPPSolver`: 单点定位求解器（带Klobuchar电离层模型）
  - `__init__(obs_file, ephemeris_file, klobuchar_file, approx_coords, debug=False, time_system='GPS')`
  - `process_file(max_epochs=2880, debug=False)`: 处理整个观测文件
  - `solve_epoch(epoch_data, initial_coords=None, debug=False)`: 解算单个历元
  - `_compute_ionosphere_delay()`: 计算Klobuchar电离层延迟

**输入参数**：

- 同 `single_point_positioning_no_ionosphere.py`
- 额外参数：`klobuchar_file`: Klobuchar电离层参数CSV文件路径

**输出结果**：

- 同 `single_point_positioning_no_ionosphere.py`

### 4. 批处理脚本

- `batch_process.py`: 通用批处理脚本
- `batch_positioning.py`: 批量处理所有测站完整时段数据
- `batch_positioning_afternoon.py`: 批量处理中午时段（12:00-14:00）数据

## 算法原理

### 单点定位基本方程

$$
P = \rho + c \cdot dt_r - c \cdot dt_s + I + T + \varepsilon
$$

其中：

- $P$: 伪距观测值
- $\rho$: 卫星到接收机的几何距离
- $c$: 光速
- $dt_r$: 接收机钟差
- $dt_s$: 卫星钟差
- $I$: 电离层延迟
- $T$: 对流层延迟
- $\varepsilon$: 测量噪声

### Klobuchar电离层模型

Klobuchar模型是GPS广播星历中提供的电离层延迟模型：

$$
I = A_1 + A_2 \cdot \cos\left(\frac{2\pi(t - A_3)}{A_4}\right)
$$

其中 $A_1, A_2, A_3, A_4$ 是从广播星历中获取的电离层参数。

### 对流层模型

本实现使用简化的对流层延迟模型：

$$
T = \frac{0.002277}{\sin(E)} \cdot P
$$

其中 $E$ 是卫星高度角，$P$ 是测站气压（假设为1013.25 hPa）。

## 使用方法

### 1. 解析RINEX文件

```python
from parse_rinex_obs import RinexObsParser

parser = RinexObsParser('obs_file.rnx')
results = parser.parse()

# 生成报告
from parse_rinex_obs import format_readable_output, format_metadata, format_observation_data
print(format_readable_output(results))
```

### 2. 单点定位（无电离层改正）

```python
from single_point_positioning_no_ionosphere import SPPSolver

approx_coords = [486854.546000, 2285099.292400, -5914955.713600]
solver = SPPSolver(
    obs_file='obs_data.csv',
    ephemeris_file='ephemeris.csv',
    approx_coords=approx_coords,
    debug=False,
    time_system='GPS'
)
results = solver.process_file(max_epochs=2880)
```

### 3. 单点定位（带Klobuchar电离层改正）

```python
from single_point_positioning_klobuchar_numpy import SPPSolver

approx_coords = [486854.546000, 2285099.292400, -5914955.713600]
solver = SPPSolver(
    obs_file='obs_data.csv',
    ephemeris_file='ephemeris.csv',
    klobuchar_file='klobuchar.csv',
    approx_coords=approx_coords,
    debug=False,
    time_system='GPS'
)
results = solver.process_file(max_epochs=2880)
```

### 4. 批处理

运行批量定位脚本：

```bash
cd code/
python batch_positioning.py        # 处理完整时段数据
python batch_positioning_afternoon.py  # 处理中午时段数据
```

## 输入文件格式

### 观测数据CSV

```csv
epoch_time,year,month,day,hour,minute,second,gps_week,gps_week_seconds,prn,c1_meters,s1_dBHz
2024-05-08 00:00:00.000000,2024,5,8,0,0,0.0,2313,0.0,1,22000000.0,45.0
...
```

### 星历CSV

包含卫星广播星历参数：PRN、TOE、卫星钟差参数、轨道根数等。

### Klobuchar参数CSV

包含电离层模型参数：$A_1, A_2, A_3, A_4$。

## 输出文件格式

定位结果文件 `{station}_{gps_week}_spp_{model}.txt`：

```
GPS Single Point Positioning Results (with Klobuchar Ionosphere Model)
==================================================================================================================================
Station: DAV100ATA
Approx Position: X=486854.5460, Y=2285099.2924, Z=-5914955.7136
GPS Week Day: 129

Epoch                             X (m)           Y (m)           Z (m)     dX (m)     dY (m)     dZ (m)   Clock (us) Satellites    RMS (m)  Max Res (m) Status                  
----------------------------------------------------------------------------------------------------------------------------------
2024-05-08 00:00:0.000000     486862.9322    2285062.3030   -5914933.8621     8.3862   -36.9894    21.8515        -0.09          8    16.6960      32.2093 Success                 
...
```

## 技术特点

1. **卫星筛选**：只处理GPS卫星（PRN 1-32），支持卫星号字符串格式（如G11、G23）
2. **伪距质量控制**：C1C伪距范围检查（1e7 ~ 5e7米），超出范围的观测值自动跳过
3. **时间系统处理**：支持GPS和UTC时间系统，自动处理闰秒（2024年GPS-UTC=18秒）
4. **调试模式**：开启debug模式可输出详细的中间计算结果，便于算法调试
5. **结果统计**：输出每个历元的卫星数、残差RMS、最大残差等质量指标
6. **粗差剔除**：基于残差分析剔除异常观测值，提高定位精度

## 依赖库

- Python 3.8+
- numpy: 矩阵运算和数值计算
- pandas: CSV文件读写

## 测站坐标

内置6个测站的近似坐标（ITRF XYZ，单位：米）：

| 测站      | X              | Y              | Z              |
| --------- | -------------- | -------------- | -------------- |
| DAV100ATA | 486854.546000  | 2285099.292400 | -5914955.713600|
| HOB200AUS | -3950072.249700| 2522415.361800 | -4311637.402200|
| KOUR00GUF | 3839591.433200 | -5059567.551400| 579956.916400  |
| NTUS00SGP | -1508022.572200| 6195577.395200 | 148799.391200  |
| NYA200NOR | 1202379.310000 | 252474.654300  | 6237786.541700 |
| WTZR00DEU | 4075580.886300 | 931853.578400  | 4801567.970700 |

## 注意事项

1. 本实现为单频GPS单点定位，仅使用C1C伪距观测值
2. 建议使用至少4颗卫星进行定位解算
3. 电离层延迟对单频定位影响较大，建议使用Klobuchar模型版本
4. 输入文件时间系统需正确设置，否则会影响卫星位置计算精度
5. 测站近似坐标需准确设置，否则会影响定位收敛速度和精度

---

**作者**：Lemon_insight  
**日期**：2026年5月  
**版本**：1.0