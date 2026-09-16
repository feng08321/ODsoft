# AOD 反演系统 (AOD Inversion System)

基于高光谱太阳直接辐照度（DNI）观测反演气溶胶光学厚度（AOD）、Ångström 参数、
936nm 水汽光学厚度（WVOD）与可降水量（PWV）。

算法依据 **QX/T 69-2024《气溶胶光学厚度 太阳光度计法》**（公式 1–14）及
《气溶胶光学厚度和 Ångström 参数反演理论基础》，详见 `docs/` 与系统内帮助页。

## 主要功能

- **AOD 反演**：7 个特征波长（340/380/400/440/500/675/870nm）时间序列 + 全波段 AOD 谱
- **Langley 定标**：由当日数据回归大气顶层信号 E0/V0，含云/扰动剔除与时段收缩，
  分上午/下午段自动择优，定标失败自动回退 Wehrli (1985) 标准光谱
- **水汽反演**：936nm 基线法（QX/T 公式 8–10）WVOD，经验公式转 PWV（mm，系数可调）
- **可视化**：AOD 时间序列、AOD 谱（线性/对数切换）、Langley 拟合诊断图、
  原始光谱与辐照度时间序列查看页
- **双速模式**：向量化快速计算 / 逐分钟慢速计算，可同图对比
- 多算法选项：大气质量数（Kasten 等 4 种）、瑞利公式（QX/T 公式 7 / Hansen & Travis）、
  气体扣除开关、水汽基线方法

## 运行

```bat
pip install -r requirements.txt
start_all.bat
```

浏览器打开 http://localhost:8000 ，上传单日 svd 格式 Excel 数据文件即可。

## 文件说明

| 文件 | 说明 |
|---|---|
| `aod_inversion.py` | 核心算法库（太阳几何、光学厚度反演、Langley 定标、WVOD/PWV） |
| `process_dni_data.py` | 数据读取与批处理（慢速逐分钟 / 快速向量化） |
| `main.py` | FastAPI 后端（异步任务 + 进度查询 + Excel 导出） |
| `frontend_english.html` | Web 前端（ECharts） |
| `help.html` | 反演原理与算法说明页 |
| `recompute_all.py` | 多日数据批量重算脚本 |

## 已知限制

- 臭氧反演模块结果不可用（待重做，紫外段差分截面与定标需完善）
- 臭氧 Chappuis 带（450–750nm）未扣除，675nm AOD 略偏高
- 水汽/氧气强吸收带内（约 690–730、760、790–870、920–980nm）的光谱数据不可定量使用
