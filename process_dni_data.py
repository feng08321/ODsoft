import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import aod_inversion
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# AOD时间序列反演的特征波长（含550/780等常用波长；936/1020处于水汽吸收带，表观AOD偏高）
AOD_KEY_WAVELENGTHS = [340, 380, 400, 440, 500, 550, 675, 780, 870, 936, 1020]

# 读取Excel文件
def read_dni_data(file_path, original_filename=None, data_start=None, data_end=None,
                  wl_start=None, wl_end=None):
    """
    读取DNI光谱数据

    Parameters
    ----------
    file_path : str
        Excel文件路径
    original_filename : str, optional
        原始文件名（如果提供，将从这里提取日期）
    data_start, data_end : str, optional
        数据起始/结束时刻 'HH:MM'，默认 04:00 起、1 分钟间隔
    wl_start, wl_end : float, optional
        数据波长范围 (nm)，默认 300-1100
        
    Returns
    -------
    dict
        包含时间、波长和辐照度数据的字典
    """
    # 读取Excel文件（calamine引擎：比openpyxl快10倍以上，低配机器上读全光谱文件从十几秒降到约1秒）
    # header=None：数据规范为无表头纯数值矩阵（首行=起始时刻数据）；若首行是文字表头则丢弃
    df = pd.read_excel(file_path, sheet_name=0, engine='calamine', header=None)
    try:
        df.iloc[0].values.astype(float)
    except (ValueError, TypeError):
        df = df.iloc[1:].reset_index(drop=True)
        print("检测到文字表头行，已丢弃")
    
    # 提取辐照度数据
    irradiance_data = df.values
    
    # 获取实际数据形状
    num_rows, num_cols = irradiance_data.shape
    print(f"实际数据形状: {num_rows}行 × {num_cols}列")
    
    # 从文件名中提取日期
    import os
    
    # 优先使用原始文件名
    if original_filename:
        filename = original_filename
        print(f"使用原始文件名提取日期: {filename}")
    else:
        filename = os.path.basename(file_path)
        print(f"使用文件路径提取日期: {filename}")
    
    # 假设文件名格式为 "20211031svd.xlsx"，前8位是日期
    try:
        date_str = filename[:8]
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        start_time = datetime(year, month, day, 4, 0, 0)
        print(f"从文件名提取日期: {start_time.strftime('%Y-%m-%d')}")
    except Exception as e:
        # 如果提取失败，使用默认日期
        start_time = datetime(2021, 10, 31, 4, 0, 0)
        print(f"无法从文件名提取日期，使用默认日期: 2021-10-31, 错误: {str(e)}")
    
    # 生成时间序列：默认 4:00 起、1 分钟间隔；
    # 提供 data_start（HH:MM）覆盖起始时刻，提供 data_end 时按 (起~止)/(行数-1) 均分间隔
    if data_start:
        try:
            hh, mm = map(int, str(data_start).strip().split(':'))
            start_time = start_time.replace(hour=hh, minute=mm, second=0, microsecond=0)
        except (ValueError, AttributeError):
            print(f"data_start 格式无效({data_start})，仍用 04:00")
    if data_end and num_rows > 1:
        try:
            eh, em = map(int, str(data_end).strip().split(':'))
            end_time = start_time.replace(hour=eh, minute=em)
            step = (end_time - start_time) / (num_rows - 1)
        except (ValueError, AttributeError):
            print(f"data_end 格式无效({data_end})，按1分钟间隔")
            step = timedelta(minutes=1)
    else:
        step = timedelta(minutes=1)
    time_series = [start_time + i * step for i in range(num_rows)]

    # 生成波长序列：默认 300-1100nm；提供 wl_start/wl_end 时按范围均分
    try:
        lo = float(wl_start) if str(wl_start or '').strip() else 300.0
        hi = float(wl_end) if str(wl_end or '').strip() else 1100.0
    except (ValueError, TypeError):
        print(f"波长范围无效({wl_start}~{wl_end})，回退 300-1100nm")
        lo, hi = 300.0, 1100.0
    wavelengths = np.linspace(lo, hi, num_cols)
    
    return {
        'time_series': time_series,
        'wavelengths': wavelengths,
        'irradiance': irradiance_data
    }

# ============ svg 水平总辐射光谱（只绘图与积分，不做反演） ============
# 按列数识别谱段（1nm 采样）：801列=300-1100nm，121列=280-400nm，751列=950-1700nm
_SVG_RANGE_BY_COLS = {801: (300.0, 1100.0), 121: (280.0, 400.0), 751: (950.0, 1700.0)}

def read_svg_data(file_path, original_filename=None, data_start=None, data_end=None,
                  wl_start=None, wl_end=None):
    """
    读取svg水平总辐射光谱数据（格式与svd一致：无表头纯数值，每行一个分钟时刻）。

    谱段：提供 wl_start/wl_end 时按该范围均分；否则按列数自动识别
    （801列→300-1100nm，121列→280-400nm，751列→950-1700nm），其余回退 300-1100nm。
    时段：默认 04:00 起 1 分钟间隔；data_start/data_end 可覆盖。
    日期从文件名前8位(YYYYMMDD)提取。

    Returns
    -------
    dict
        {'time_series', 'wavelengths', 'irradiance'}
    """
    df = pd.read_excel(file_path, sheet_name=0, engine='calamine', header=None)
    # 首行若为文字表头则丢弃（数据规范为无表头纯数值矩阵，首行=起始时刻数据）
    try:
        df.iloc[0].values.astype(float)
    except (ValueError, TypeError):
        df = df.iloc[1:].reset_index(drop=True)
        print("检测到文字表头行，已丢弃")
    irradiance_data = df.values.astype(float)
    num_rows, num_cols = irradiance_data.shape
    print(f"svg 数据形状: {num_rows}行 × {num_cols}列")

    import os
    filename = original_filename if original_filename else os.path.basename(file_path)
    try:
        date_str = filename[:8]
        start_time = datetime(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]), 4, 0, 0)
        print(f"从文件名提取日期: {start_time.strftime('%Y-%m-%d')}")
    except Exception as e:
        start_time = datetime(2021, 10, 31, 4, 0, 0)
        print(f"无法从文件名提取日期，使用默认日期: 2021-10-31, 错误: {str(e)}")

    # 时间轴：默认 04:00 起 1 分钟间隔；data_start/data_end 可覆盖（与 read_dni_data 同规则）
    if data_start:
        try:
            hh, mm = map(int, str(data_start).strip().split(':'))
            start_time = start_time.replace(hour=hh, minute=mm, second=0, microsecond=0)
        except (ValueError, AttributeError):
            print(f"data_start 格式无效({data_start})，仍用 04:00")
    if data_end and num_rows > 1:
        try:
            eh, em = map(int, str(data_end).strip().split(':'))
            end_time = start_time.replace(hour=eh, minute=em)
            step = (end_time - start_time) / (num_rows - 1)
        except (ValueError, AttributeError):
            step = timedelta(minutes=1)
    else:
        step = timedelta(minutes=1)
    time_series = [start_time + i * step for i in range(num_rows)]
    # 波长：用户指定范围优先；否则按列数自动识别
    try:
        user_range = str(wl_start or '').strip() and str(wl_end or '').strip()
        if user_range:
            wl_lo, wl_hi = float(wl_start), float(wl_end)
        else:
            wl_lo, wl_hi = _SVG_RANGE_BY_COLS.get(num_cols, (300.0, 1100.0))
    except (ValueError, TypeError):
        print(f"波长范围无效({wl_start}~{wl_end})，回退按列数识别")
        wl_lo, wl_hi = _SVG_RANGE_BY_COLS.get(num_cols, (300.0, 1100.0))
    else:
        if user_range:
            print(f"svg 谱段使用用户指定: {wl_lo:.0f}-{wl_hi:.0f}nm")
    wavelengths = np.linspace(wl_lo, wl_hi, num_cols)
    print(f"svg 谱段识别: {wl_lo:.0f}-{wl_hi:.0f}nm ({num_cols}列)")

    return {'time_series': time_series, 'wavelengths': wavelengths, 'irradiance': irradiance_data}

def apply_exclude_periods(data, periods):
    """
    按时段剔除数据：将处于任一剔除时段内的整行辐照度置为 NaN
    （NaN 被后续 E>0 过滤、积分与绘图自然视为无效/断点，不参与任何计算）。

    Parameters
    ----------
    data : dict
        read_dni_data / read_svg_data 的返回（原地修改其 irradiance）
    periods : list[list[str]]
        剔除时段 [["HH:MM","HH:MM"], ...]；自动裁剪到数据实际时间范围，
        起始>=结束或完全落在数据范围外的时段忽略

    Returns
    -------
    list[list[str]]
        实际生效的时段（裁剪后），用于结果区展示
    """
    if not periods:
        return []
    times = data['time_series']
    if not times:
        return []
    irr = np.asarray(data['irradiance'], dtype=float)
    day = times[0].date()
    t_start, t_end = times[0], times[-1]
    t_min = np.array([t.hour * 60 + t.minute for t in times])
    mask = np.zeros(len(times), dtype=bool)
    applied = []
    for p in periods:
        try:
            s = datetime.strptime(f'{day} {p[0]}', '%Y-%m-%d %H:%M')
            e = datetime.strptime(f'{day} {p[1]}', '%Y-%m-%d %H:%M')
        except Exception:
            continue
        s = max(s, t_start)
        e = min(e, t_end)
        if s >= e:
            continue
        s_min = s.hour * 60 + s.minute
        e_min = e.hour * 60 + e.minute
        mask |= (t_min >= s_min) & (t_min <= e_min)
        applied.append([s.strftime('%H:%M'), e.strftime('%H:%M')])
    if mask.any():
        irr[mask] = np.nan
        data['irradiance'] = irr
        print(f"剔除时段生效 {applied}，共剔除 {int(mask.sum())} 行")
    return applied

# CIE 1931 明视觉光谱光视效率 V(λ)，380-780nm，5nm 间隔
_CIE_V_WL = np.arange(380.0, 781.0, 5.0)
_CIE_V = np.array([
    0.000039, 0.000064, 0.000120, 0.000217, 0.000396, 0.000640, 0.001210, 0.002180,
    0.004000, 0.007300, 0.011600, 0.016840, 0.023000, 0.029800, 0.038000, 0.048000,
    0.060000, 0.073900, 0.090980, 0.112600, 0.139020, 0.169300, 0.208020, 0.258600,
    0.323000, 0.407300, 0.503000, 0.608200, 0.710000, 0.793200, 0.862000, 0.914850,
    0.954000, 0.980300, 0.994950, 1.000000, 0.995000, 0.978600, 0.952000, 0.915400,
    0.870000, 0.816300, 0.757000, 0.694900, 0.631000, 0.566800, 0.503000, 0.441200,
    0.381000, 0.321000, 0.265000, 0.217000, 0.175000, 0.138200, 0.107000, 0.081600,
    0.061000, 0.044580, 0.032000, 0.023200, 0.017000, 0.011920, 0.008210, 0.005723,
    0.004102, 0.002929, 0.002091, 0.001484, 0.001047, 0.000740, 0.000520, 0.000361,
    0.000249, 0.000172, 0.000120, 0.000085, 0.000060, 0.000042, 0.000030, 0.000021,
    0.000015])

def compute_svg_integrals(wavelengths, irradiance):
    """
    对 svg 光谱按时刻做波长积分（辐照度单位 W/(m²·nm)，向量化实现）。

    积分项（数据谱段未完整覆盖的波段结果为 NaN）：
      total    — 全谱段积分辐照度 (W/m²)
      uv       — 300-400nm 紫外积分 (W/m²)
      vis      — 400-700nm 可见积分 (W/m²)
      nir      — 700-1100nm 近红外积分 (W/m²)
      par_w    — PAR 光合有效辐射 400-700nm 能量积分 (W/m²)
      par_ppfd — PAR 光子通量密度 (μmol/(m²·s))，PPFD = ∫E(λ)·λ·1e-9/(h·c·N_A)dλ ×1e6
      lux      — 照度 (lx)，L = 683·∫E(λ)·V(λ)dλ，V(λ) 为 CIE 1931 明视觉曲线

    Returns
    -------
    dict[str, np.ndarray]  每个键对应与时刻数等长的 float 数组
    """
    wl = np.asarray(wavelengths, dtype=float)
    irr = np.asarray(irradiance, dtype=float)
    n = irr.shape[0]
    out = {k: np.full(n, np.nan) for k in ('total', 'uv', 'vis', 'nir', 'par_w', 'par_ppfd', 'lux')}

    # 孤立坏列（如328/349nm）沿波长方向线性插值填补，避免积分因单个NaN整行作废；
    # 整行全NaN的时刻（夜间）插值后仍为NaN，结果自然保留NaN
    irr = pd.DataFrame(irr).interpolate(axis=1, limit_direction='both').values

    def _band(lo, hi):
        """返回覆盖 [lo,hi] 的列切片；谱段未完整覆盖时返回 None"""
        if wl[0] > lo + 1e-6 or wl[-1] < hi - 1e-6:
            return None
        i0 = int(np.searchsorted(wl, lo))
        i1 = int(np.searchsorted(wl, hi, side='right'))
        return slice(i0, max(i1, i0 + 2))

    out['total'] = np.trapezoid(irr, wl, axis=1)
    for key, lo, hi in (('uv', 300, 400), ('vis', 400, 700), ('nir', 700, 1100), ('par_w', 400, 700)):
        s = _band(lo, hi)
        if s is not None:
            out[key] = np.trapezoid(irr[:, s], wl[s], axis=1)

    # PPFD：h·c·N_A ≈ 0.119626 J·m/mol
    s = _band(400, 700)
    if s is not None:
        photon_flux = np.trapezoid(irr[:, s] * wl[s] * 1e-9, wl[s], axis=1)  # 光子/(s·m²)
        out['par_ppfd'] = photon_flux / 0.119626 * 1e6

    # 照度：V(λ) 插值到数据波长，380-780nm 之外按 0 计
    s = _band(380, 780)
    if s is not None:
        v = np.interp(wl[s], _CIE_V_WL, _CIE_V)
        out['lux'] = 683.0 * np.trapezoid(irr[:, s] * v, wl[s], axis=1)

    return out


# 处理DNI数据，反演AOD
def process_dni_data(dni_data, latitude=49.0, longitude=119.4, pressure=840.0,
                     airmass_method='kasten', rayleigh_method='standard',
                     gas_correction=False, e0_langley=None, progress_cb=None):  # 1.5km海拔的气压约为840 hPa
    """
    处理DNI数据，反演AOD
    
    Parameters
    ----------
    dni_data : dict
        DNI数据
    latitude : float
        纬度
    longitude : float
        经度
    pressure : float
        地面气压 (hPa)
        
    Returns
    -------
    pd.DataFrame
        AOD反演结果
    """
    time_series = dni_data['time_series']
    wavelengths = dni_data['wavelengths']
    irradiance_data = dni_data['irradiance']
    
    results = []

    # Langley定标的E0（如有）覆盖Wehrli默认值
    e0_ref = None
    if e0_langley:
        e0_ref = dict(aod_inversion.WEHRLI_1985)
        e0_ref.update(e0_langley)

    # 对每个时间点进行处理
    total_time_points = len(time_series)
    for i, timestamp in enumerate(time_series):
        # 显示进度
        if i % 10 == 0:
            print(f"处理进度: {i}/{total_time_points} ({i/total_time_points*100:.1f}%)")
            if progress_cb:
                progress_cb(i / total_time_points)

        # 获取该时间点的辐照度数据
        irradiances = irradiance_data[i, :]

        # 计算太阳天顶角
        try:
            solar_zenith = aod_inversion.solar_zenith_angle(latitude, longitude, timestamp)
        except Exception as e:
            print(f"计算太阳天顶角失败: {str(e)}")
            solar_zenith = 90.0  # 设置为90度，这样会跳过该时间点
        
        # 检查太阳天顶角是否有效（太阳天顶角必须小于85度才能进行AOD反演）
        if solar_zenith >= 85:
            # 太阳天顶角无效，跳过该时间点
            row = {'datetime': timestamp, 'solar_zenith': solar_zenith}
            for wl in AOD_KEY_WAVELENGTHS:
                row[f'aod_{wl}'] = np.nan
            row['angstrom_alpha'] = np.nan
            results.append(row)
            continue

        # 对每个波长进行AOD反演
        aod_results = {}
        # 处理更多关键波长
        key_wavelengths = AOD_KEY_WAVELENGTHS
        for wavelength in key_wavelengths:
            # 找到最接近的波长索引
            idx = np.argmin(np.abs(wavelengths - wavelength))
            irradiance = irradiances[idx]
            
            # 检查辐照度是否有效（必须大于0）
            if irradiance <= 0:
                aod_results[wavelength] = np.nan
                continue
            
            # 反演AOD
            try:
                result = aod_inversion.invert_aod(
                    wavelength=wavelength,
                    irradiance=irradiance,
                    solar_zenith=solar_zenith,
                    date=timestamp,
                    pressure=pressure,
                    ozone_du=300.0,
                    e0_reference=e0_ref,
                    airmass_method=airmass_method,
                    rayleigh_method=rayleigh_method,
                    gas_correction=gas_correction
                )
                aod_results[wavelength] = result['aod']
            except Exception as e:
                aod_results[wavelength] = np.nan
        
        # 计算Angstrom指数（440-870nm）
        if 440 in aod_results and 870 in aod_results:
            try:
                alpha = aod_inversion.calculate_angstrom_exponent(aod_results, 440, 870)
            except:
                alpha = np.nan
        else:
            alpha = np.nan
        
        # 保存结果（按 AOD_KEY_WAVELENGTHS 动态生成列）
        row = {'datetime': timestamp, 'solar_zenith': solar_zenith}
        for wl in key_wavelengths:
            row[f'aod_{wl}'] = aod_results.get(wl, np.nan)
        row['angstrom_alpha'] = alpha
        results.append(row)

    return pd.DataFrame(results)

# 处理高光谱数据，计算每个波长的光学厚度（向量化版本）
def process_hyperspectral_data(dni_data, latitude=49.0, longitude=119.4, pressure=840.0, ozone_du=300.0,
                               airmass_method='kasten', rayleigh_method='standard',
                               gas_correction=False, e0_langley=None, progress_cb=None):
    """
    处理高光谱数据，计算每个波长的光学厚度（向量化实现）。

    对每个采样时刻，一次完成全部波长的 Beer-Lambert 反演：
        总光学厚度 tau = -ln(E/(E0·esd))/m
        AOD谱      tau_aero = tau - tau_R(λ) (- tau_gas(λ)，可选)

    Parameters
    ----------
    dni_data : dict
        DNI数据
    latitude, longitude, pressure : float
        站点参数与气压 (hPa)
    ozone_du : float
        臭氧柱总量 (DU)，仅 gas_correction=True 时使用
    e0_langley : dict, optional
        全波段 Langley 定标 E0 {波长: E0}（langley_calibrate_spectrum 的输出），
        未覆盖的波长回退 Wehrli 标准值（含插值）

    Returns
    -------
    dict
        {时刻'HH:MM': {'wavelengths', 'optical_depths'(总), 'aod'(气溶胶), 'solar_zenith'}}
    """
    time_series = dni_data['time_series']
    wavelengths = np.asarray(dni_data['wavelengths'], dtype=float)
    irradiance_data = np.asarray(dni_data['irradiance'], dtype=float)

    hyperspectral_data = {}

    if not time_series:
        return hyperspectral_data

    # 与波长无关的量只算一次
    e0_ref = _merge_e0(e0_langley)
    e0_vec = np.array([_e0_at(e0_ref, wl) for wl in wavelengths])
    ray_vec = np.array([_rayleigh_at(wl, pressure, rayleigh_method) for wl in wavelengths])
    gas_vec = np.zeros(len(wavelengths))
    if gas_correction:
        gas_vec = np.array([
            aod_inversion.calculate_total_absorption_optical_depth(wl, pressure=pressure)
            for wl in wavelengths
        ])

    doy = aod_inversion.calculate_day_of_year(time_series[0])
    esd = aod_inversion.earth_sun_distance_factor(doy)

    # 选择从6:00到19:00的所有时间点，每个小时一个点
    base_date = time_series[0].date()
    target_times = [f'{hour:02d}:00' for hour in range(6, 20)]  # 6:00到19:00
    ts_sec = np.array([(t - time_series[0]).total_seconds() for t in time_series])

    for ti, target_time in enumerate(target_times):
        if progress_cb:
            progress_cb(ti / len(target_times))
        target_datetime = pd.to_datetime(f'{base_date} {target_time}')
        closest_idx = int(np.argmin(np.abs(ts_sec - (target_datetime - time_series[0]).total_seconds())))
        timestamp = time_series[closest_idx]

        # 计算太阳天顶角
        try:
            solar_zenith = aod_inversion.solar_zenith_angle(latitude, longitude, timestamp)
        except Exception:
            continue
        if solar_zenith >= 85:
            continue
        airmass = aod_inversion.calculate_airmass(solar_zenith, method=airmass_method)
        if not np.isfinite(airmass) or airmass <= 0:
            continue

        # 向量化反演：全部波长一次完成
        E = irradiance_data[closest_idx, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            total_od = np.where(E > 0, -np.log(E / (e0_vec * esd)) / airmass, np.nan)
        aod_spec = total_od - ray_vec - gas_vec

        hyperspectral_data[timestamp.strftime('%H:%M')] = {
            'wavelengths': wavelengths,
            'optical_depths': total_od.tolist(),
            'aod': aod_spec.tolist(),
            'solar_zenith': solar_zenith
        }

    return hyperspectral_data

# 绘制AOD时间序列图
def plot_aod_time_series(aod_results):
    """
    绘制AOD时间序列图
    
    Parameters
    ----------
    aod_results : pd.DataFrame
        AOD反演结果
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    plt.figure(figsize=(12, 6))
    
    # 绘制不同波长的AOD
    plt.plot(aod_results['datetime'], aod_results['aod_440'], label='AOD 440nm', alpha=0.7)
    plt.plot(aod_results['datetime'], aod_results['aod_500'], label='AOD 500nm', alpha=0.7)
    plt.plot(aod_results['datetime'], aod_results['aod_675'], label='AOD 675nm', alpha=0.7)
    plt.plot(aod_results['datetime'], aod_results['aod_870'], label='AOD 870nm', alpha=0.7)
    
    # 设置图表属性
    plt.title('2021年10月31日AOD时间序列', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('AOD', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # 格式化x轴日期
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.xticks(rotation=45)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('20211031_aod_time_series.png', dpi=300)
    print("AOD时间序列图已保存: 20211031_aod_time_series.png")
    
    # 关闭图表，避免阻塞
    plt.close()
    # plt.show()

# 计算日平均AOD
def calculate_daily_average(aod_results):
    """
    计算日平均AOD
    
    Parameters
    ----------
    aod_results : pd.DataFrame
        AOD反演结果
        
    Returns
    -------
    dict
        日平均AOD结果
    """
    # 计算各波长的日平均AOD
    daily_avg = {
        'aod_440': aod_results['aod_440'].mean(),
        'aod_500': aod_results['aod_500'].mean(),
        'aod_675': aod_results['aod_675'].mean(),
        'aod_870': aod_results['aod_870'].mean(),
        'angstrom_alpha': aod_results['angstrom_alpha'].mean()
    }
    
    return daily_avg

def invert_ozone(dni_data, latitude=49.0, longitude=119.4, pressure=840.0):
    """
    反演臭氧柱总量 (DOAS方法)
    
    基于差分光学吸收光谱法(DOAS)反演臭氧柱浓度
    
    Parameters
    ----------
    dni_data : dict
        DNI数据
    latitude : float
        纬度
    longitude : float
        经度
    pressure : float
        地面气压 (hPa)
        
    Returns
    -------
    pd.DataFrame
        臭氧反演结果
    """
    time_series = dni_data['time_series']
    wavelengths = dni_data['wavelengths']
    irradiance_data = dni_data['irradiance']
    
    results = []
    
    # DOAS拟合使用的波长范围
    wavelength_range = (305.0, 380.0)
    
    # 对每个时间点进行处理
    total_time_points = len(time_series)
    for i, timestamp in enumerate(time_series):
        # 显示进度
        if i % 10 == 0:
            print(f"处理进度: {i}/{total_time_points} ({i/total_time_points*100:.1f}%)")
        
        # 获取该时间点的辐照度数据
        irradiances = irradiance_data[i, :]
        
        # 计算太阳天顶角
        try:
            solar_zenith = aod_inversion.solar_zenith_angle(latitude, longitude, timestamp)
        except Exception as e:
            solar_zenith = np.nan
        
        # 计算大气质量数
        try:
            airmass = aod_inversion.calculate_airmass(solar_zenith)
        except Exception as e:
            airmass = np.nan
        
        # 检查大气质量数是否有效（太阳天顶角必须小于85度才能进行臭氧反演）
        if np.isnan(airmass) or np.isinf(airmass) or airmass > 10:
            # 大气质量数无效，跳过该时间点
            results.append({
                'datetime': timestamp,
                'solar_zenith': solar_zenith,
                'airmass': airmass,
                'ozone_305': np.nan,
                'ozone_310': np.nan,
                'ozone_320': np.nan,
                'ozone_340': np.nan,
                'ozone_360': np.nan,
                'ozone_380': np.nan,
                'average_ozone': np.nan
            })
            continue
        
        # 使用DOAS方法反演臭氧
        try:
            doas_result = aod_inversion.doas_ozone_retrieval(
                irradiances=irradiances,
                wavelengths=wavelengths,
                solar_zenith=solar_zenith,
                airmass=airmass,
                e0_reference=aod_inversion.WEHRLI_1985,
                pressure=pressure,
                wavelength_range=wavelength_range,
                polynomial_order=2
            )
            
            ozone_by_wavelength = doas_result.get('ozone_by_wavelength', {})
            avg_ozone = doas_result.get('vcd_ozone', np.nan)
            
        except Exception as e:
            print(f"DOAS反演失败: {str(e)}")
            ozone_by_wavelength = {}
            avg_ozone = np.nan
        
        # 保存结果
        results.append({
            'datetime': timestamp,
            'solar_zenith': solar_zenith,
            'airmass': airmass,
            'ozone_305': ozone_by_wavelength.get(305, np.nan),
            'ozone_310': ozone_by_wavelength.get(310, np.nan),
            'ozone_320': ozone_by_wavelength.get(320, np.nan),
            'ozone_340': ozone_by_wavelength.get(340, np.nan),
            'ozone_360': ozone_by_wavelength.get(360, np.nan),
            'ozone_380': ozone_by_wavelength.get(380, np.nan),
            'average_ozone': avg_ozone
        })
    
    return pd.DataFrame(results)

# 按天Langley定标：回归大气顶层信号E0
def langley_calibrate_day(dni_data, latitude=49.0, longitude=119.4,
                          wavelengths_cal=None, segment='auto',
                          m_min=2.0, m_max=6.0, min_r2=0.9,
                          cloud_screen=True, shrink=True, max_depth=2, time_range=None,
                          sigma_e0_max=0.01, progress_cb=None, return_plot=False):
    """
    对一天的数据做Langley定标，回归各波长的大气顶层信号 E0。

    流程：
    1. 计算每个时间点的太阳天顶角、大气质量数、日地距离修正因子
    2. 按大气质量数最小值（正午）分为上午/下午两段
    3. 数据清理：基于表观光学厚度时间稳定性剔除云遮挡/大气剧烈扰动点
       （cloud_screen=True 时启用）
    4. 每个波长分别对两段做 ln(E/esd)~m 线性回归（Bouguer-Langley迭代剔除）；
       整段拟合不达标时自动二分缩小时段重试（shrink=True 时启用）
    5. segment='auto' 时选取 R² 较高的半日段

    Parameters
    ----------
    dni_data : dict
        DNI数据
    latitude, longitude : float
        观测点经纬度
    wavelengths_cal : list, optional
        需要定标的波长（nm），默认 [340,380,400,440,500,675,870,936,1020]
    segment : str
        拟合时段: 'auto'(日级统一时段：上午优先，上午整体不达标才统一用下午，
        全部波长使用同一时段), 'morning', 'afternoon', 'all'(全天)。
        除人工 time_range 外，全部波长在统一时段内还使用统一的拟合时间窗
        （从各波长收缩窗口中日级评选公共窗口，保证 E0 光谱的时间一致性，
        统一性优先于单波长最高R²）
    time_range : tuple/list[str], optional
        人工定标时段 ('HH:MM','HH:MM')，给定后覆盖 segment 分段，仅在该时段内拟合
    m_min, m_max : float
        参与拟合的大气质量数范围
    min_r2 : float
        可接受的最低 R²
    cloud_screen : bool
        是否启用基于光学厚度稳定性的云/扰动清理
    shrink : bool
        整段拟合不达标时是否自动缩小时段范围
    max_depth : int
        时段收缩最大层数（2=最短缩到半日的1/4）

    Returns
    -------
    tuple (e0_dict, report_df)
        e0_dict : {波长: E0} 仅含定标成功的波长
        report_df : 各波长各半日段的定标详情（e0, tau_mean, r2, n_points,
                    segment, success, depth, t_start, t_end, n_clear,
                    day_segment=日级统一时段, selected=该行是否提供了E0）
    """
    if wavelengths_cal is None:
        wavelengths_cal = [340, 380, 400, 440, 500, 675, 870, 936, 1020]

    time_series = dni_data['time_series']
    wavelengths = dni_data['wavelengths']
    irradiance_data = dni_data['irradiance']

    n = len(time_series)
    airmasses = np.full(n, np.nan)
    for i, ts in enumerate(time_series):
        try:
            sza = aod_inversion.solar_zenith_angle(latitude, longitude, ts)
            airmasses[i] = aod_inversion.calculate_airmass(sza)
        except Exception:
            pass

    doy = aod_inversion.calculate_day_of_year(time_series[0])
    esd = aod_inversion.earth_sun_distance_factor(doy)

    # 正午分界（大气质量数最小处）
    valid_m = np.isfinite(airmasses)
    if valid_m.sum() < 20:
        return {}, pd.DataFrame()
    idx_noon = int(np.nanargmin(airmasses))
    segments = {
        'morning': np.arange(0, idx_noon + 1),
        'afternoon': np.arange(idx_noon, n)
    }

    # 人工指定定标时段（HH:MM~HH:MM）覆盖上/下午分段
    if time_range:
        try:
            t_min_arr = np.array([t.hour * 60 + t.minute for t in time_series])
            sh, sm = map(int, str(time_range[0]).split(':'))
            eh, em = map(int, str(time_range[1]).split(':'))
            if sh * 60 + sm < eh * 60 + em:
                manual = np.where((t_min_arr >= sh * 60 + sm) & (t_min_arr <= eh * 60 + em))[0]
                if len(manual) >= 10:
                    segments = {'manual': manual}
        except Exception:
            pass

    # ---- 第一遍：各波长分别拟合全部候选时段 ----
    per_wl = []
    for wi, wl in enumerate(wavelengths_cal):
        if progress_cb:
            progress_cb(wi / len(wavelengths_cal))
        idx = int(np.argmin(np.abs(wavelengths - wl)))
        signals = irradiance_data[:, idx]

        # 数据清理：剔除云遮挡/剧烈扰动点
        if cloud_screen:
            clear = aod_inversion.cloud_screen_variability(airmasses, signals, esd)
        else:
            clear = np.ones(n, dtype=bool)
        n_clear = int(clear.sum())

        def fit_seg(seg_idx):
            if shrink:
                return aod_inversion.langley_calibration_shrinking(
                    airmasses[seg_idx], signals[seg_idx], esd,
                    timestamps=[time_series[i] for i in seg_idx],
                    clear_mask=clear[seg_idx],
                    m_min=m_min, m_max=m_max, min_r2=min_r2,
                    sigma_e0_max=sigma_e0_max,
                    max_depth=max_depth, return_plot=return_plot)
            else:
                res = aod_inversion.langley_calibration(
                    airmasses[seg_idx][clear[seg_idx]], signals[seg_idx][clear[seg_idx]],
                    esd, m_min=m_min, m_max=m_max, min_r2=min_r2,
                    sigma_e0_max=sigma_e0_max)
                res['depth'] = 0
                return res

        seg_results = {}
        for seg_name, seg_idx in segments.items():
            res = fit_seg(seg_idx)
            res['segment'] = seg_name
            seg_results[seg_name] = res
        if segment == 'all':
            res_all = fit_seg(np.arange(n))
            res_all['segment'] = 'all'
            seg_results['all'] = res_all
        per_wl.append({'wl': wl, 'signals': signals, 'clear': clear, 'n_clear': n_clear,
                       'seg_results': seg_results})

    # ---- 日级统一选段：同一时段应用于全部波长，保持 E0 光谱一致 ----
    # auto 模式上午优先：上午达标波长数不少于下午则统一用上午；
    # 上午整体不达标（如上午阴雨、下午放晴）才统一用下午
    if 'manual' in segments:
        day_segment = 'manual'
    elif segment in ('morning', 'afternoon', 'all'):
        day_segment = segment
    else:  # auto
        n_am = sum(1 for it in per_wl if it['seg_results'].get('morning', {}).get('success'))
        n_pm = sum(1 for it in per_wl if it['seg_results'].get('afternoon', {}).get('success'))
        day_segment = 'morning' if n_am >= n_pm else 'afternoon'

    # ---- 日级统一拟合窗口：各波长收缩出的子窗口往往不同，
    # 在统一半日分支内评选对全部波长综合最优的公共窗口（达标波长数优先、
    # 中位 R² 次之），所有波长在同一窗口重拟合，保证 E0(λ) 的时间一致性。
    # 统一性优先于单波长最高 R² ----
    day_window = None
    if 'manual' not in segments:
        cand = {}
        for it in per_wl:
            r = it['seg_results'].get(day_segment)
            if r is not None and r.get('success') and r.get('t_start') is not None:
                key = (r['t_start'], r['t_end'])
                cand[key] = cand.get(key, 0) + 1
        if len(cand) > 1:
            seg_idx_all = segments[day_segment] if day_segment in segments else np.arange(n)
            seg_times = [time_series[i] for i in seg_idx_all]
            scored = []
            for t1, t2 in cand:
                win_idx = seg_idx_all[[t1 <= t <= t2 for t in seg_times]]
                n_pass, r2s = 0, []
                for it in per_wl:
                    cw = it['clear'][win_idx]
                    res_w = aod_inversion.langley_calibration(
                        airmasses[win_idx][cw], it['signals'][win_idx][cw], esd,
                        m_min=m_min, m_max=m_max, min_r2=min_r2,
                        sigma_e0_max=sigma_e0_max)
                    if res_w['success']:
                        n_pass += 1
                        r2s.append(res_w['r2'])
                scored.append((n_pass, float(np.median(r2s)) if r2s else -1.0, (t1, t2)))
            scored.sort(key=lambda x: (-x[0], -x[1]))
            day_window = scored[0][2]

    # ---- 第二遍：按统一时段生成 E0、报告与绘图数据 ----
    e0_dict = {}
    report_rows = []
    plot_details = {}
    explicit = ('manual' in segments) or segment in ('morning', 'afternoon', 'all')
    for it in per_wl:
        wl = it['wl']
        signals = it['signals']
        n_clear = it['n_clear']
        seg_results = it['seg_results']
        clear = it['clear']

        # 统一窗口重拟合：替换统一时段的逐波长收缩结果，索引范围同步替换
        it['win_idx'] = None
        if day_window is not None and day_segment in seg_results:
            t1, t2 = day_window
            seg_idx_all = segments[day_segment] if day_segment in segments else np.arange(n)
            seg_times = [time_series[i] for i in seg_idx_all]
            win_idx = seg_idx_all[[t1 <= t <= t2 for t in seg_times]]
            cw = clear[win_idx]
            out = aod_inversion.langley_calibration(
                airmasses[win_idx][cw], signals[win_idx][cw], esd,
                m_min=m_min, m_max=m_max, min_r2=min_r2,
                sigma_e0_max=sigma_e0_max, return_details=return_plot)
            if return_plot:
                res_w, det = out
            else:
                res_w, det = out, None
            res_w['segment'] = day_segment
            res_w['depth'] = 0
            res_w['t_start'] = t1
            res_w['t_end'] = t2
            if det is not None:
                used_on_win = np.zeros(len(win_idx), dtype=bool)
                used_on_win[cw] = det['used_mask']
                res_w['plot'] = {'used': used_on_win,
                                 'slope': det['slope'], 'intercept': det['intercept']}
            seg_results[day_segment] = res_w
            it['win_idx'] = win_idx

        best = None
        best_seg_idx = None
        for seg_name, res in seg_results.items():
            if explicit and seg_name != day_segment:
                continue  # 显式指定时段时报告只含该段
            report_rows.append({'wavelength': wl, 'segment': res['segment'],
                                'e0': res['e0'], 'tau_mean': res['tau_mean'],
                                'r2': res['r2'], 'rmse': res['rmse'],
                                'sigma_e0': res.get('sigma_e0'),
                                'estimated': bool(res.get('estimated', False)),
                                'n_points': res['n_points'], 'n_clear': n_clear,
                                'depth': res.get('depth', 0),
                                't_start': res.get('t_start'), 't_end': res.get('t_end'),
                                'day_segment': day_segment,
                                'selected': bool(seg_name == day_segment and res['success']),
                                'success': res['success']})
            if seg_name == day_segment and res['success']:
                best = res
                best_seg_idx = (it['win_idx'] if it['win_idx'] is not None
                                else (segments[seg_name] if seg_name in segments
                                      else np.arange(n)))

        if best is not None:
            e0_dict[wl] = best['e0']

        # 汇总绘图数据（全天 m/y，标记参与拟合的点）
        if return_plot:
            disp = best
            disp_seg_idx = best_seg_idx
            if disp is None:
                # 定标失败时优先展示统一时段的拟合用于诊断；
                # 该段无绘图数据时退而展示 R² 较高的候选段
                disp = seg_results.get(day_segment)
                disp_seg_idx = (it['win_idx'] if it['win_idx'] is not None
                                else (segments[day_segment] if day_segment in segments
                                      else np.arange(n)))
                if disp is None or disp.get('plot') is None:
                    r2_list = [(r['r2'] if np.isfinite(r['r2']) else -9, sn)
                               for sn, r in seg_results.items()]
                    j = int(np.argmax([x[0] for x in r2_list]))
                    sn = r2_list[j][1]
                    disp = seg_results[sn]
                    disp_seg_idx = segments[sn] if sn in segments else np.arange(n)
            p = disp.get('plot')
            if p is not None and disp_seg_idx is not None:
                y_full = np.full(n, np.nan)
                ok = np.isfinite(airmasses) & np.isfinite(signals) & (signals > 0)
                y_full[ok] = np.log(signals[ok] / esd)
                used_full = np.zeros(n, dtype=bool)
                used_local = np.asarray(p['used'], dtype=bool)
                used_full[disp_seg_idx[used_local]] = True
                def _fmt_hhmm(t):
                    if t is None:
                        return None
                    if hasattr(t, 'strftime'):
                        return t.strftime('%H:%M')
                    s = str(t)
                    return s.split('T')[1][:5] if 'T' in s else s[:5]
                plot_details[str(wl)] = {
                    'm': [None if not np.isfinite(v) else float(v) for v in airmasses],
                    'y': [None if not np.isfinite(v) else float(v) for v in y_full],
                    'used': used_full.tolist(),
                    'slope': p['slope'], 'intercept': p['intercept'],
                    'r2': disp['r2'], 'segment': disp['segment'],
                    't_start': _fmt_hhmm(disp.get('t_start')),
                    't_end': _fmt_hhmm(disp.get('t_end')),
                    'success': bool(disp['success']),
                    'estimated': bool(disp.get('estimated', False))
                }

    if return_plot:
        return e0_dict, pd.DataFrame(report_rows), plot_details
    return e0_dict, pd.DataFrame(report_rows)


def langley_calibrate_spectrum(dni_data, latitude=49.0, longitude=119.4,
                               airmass_method='kasten',
                               m_min=1.5, m_max=5.5, min_points=30,
                               sigma=2.0, max_iter=3, min_r2=0.85,
                               cloud_screen=True, segment='auto', tau_max=1.0,
                               sigma_e0_max=0.01, time_range=None, progress_cb=None):
    """
    全波段 Langley 定标：对数据中全部波长（~800个）回归大气顶层信号 E0。

    与 langley_calibrate_day（9个特征波长、含时段收缩）相比，本函数面向
    高光谱全波段，采用向量化云清理 + 逐波长 sigma 迭代回归，整段不达标时
    退而尝试上午/下午半日段，取 R² 较高者。仍失败的波长不进入返回字典
    （调用方应对这些波长回退 Wehrli 标准值）。

    注意：水汽/氧气强吸收带内（720/760/820/940nm 等）光学厚度随水汽含量
    变化，Langley 线性假设不成立，这些波段即使 R² 达标也不应用于定量分析。

    Parameters
    ----------
    dni_data : dict
        DNI数据（time_series / wavelengths / irradiance）
    m_min, m_max : float
        参与拟合的大气质量数范围
    min_points : int
        最少有效点数
    sigma, max_iter : float, int
        Bouguer-Langley 残差剔除阈值与迭代次数
    min_r2 : float
        可接受的最低 R²
    cloud_screen : bool
        是否启用基于表观光学厚度稳定性的云/扰动清理（向量化）
    segment : str
        拟合时段: 'auto'(日级统一时段：上午优先，上午整体不达标才统一用下午，
        全部波长使用同一时段), 'morning', 'afternoon', 'all'(全天)
    tau_max : float
        可接受的平均总光学厚度上限（默认1.0），排除云崩塌造成的高R²假拟合

    Returns
    -------
    tuple (e0_dict, report_df)
        e0_dict : {波长: E0} 仅含定标成功的波长
        report_df : 每个波长的定标详情（e0, tau_mean, r2, n_points, segment, success）
    """
    time_series = dni_data['time_series']
    wavelengths = np.asarray(dni_data['wavelengths'], dtype=float)
    irr = np.asarray(dni_data['irradiance'], dtype=float)
    n, nwl = irr.shape

    # 太阳几何全天计算一次（各波长共用）
    sza, am, esd = _precompute_geometry(time_series, latitude, longitude, airmass_method)

    # 表观总光学厚度矩阵 tau = -ln(E/esd)/m，用于云清理
    with np.errstate(divide='ignore', invalid='ignore'):
        tau_app = np.where(irr > 0, -np.log(irr / esd) / am[:, None], np.nan)

    valid_base = np.isfinite(am) & (am >= m_min) & (am <= m_max)

    if cloud_screen:
        # 相邻点跳变（向量化）
        dtau = np.abs(np.diff(tau_app, axis=0))
        jump = np.zeros_like(tau_app, dtype=bool)
        jump[:-1, :] |= dtau > 0.05
        jump[1:, :] |= dtau > 0.05
        # 滑动窗口标准差（向量化，pandas rolling）
        roll_std = pd.DataFrame(tau_app).rolling(5, center=True, min_periods=3).std().values
        clear = np.isfinite(tau_app) & ~jump & ~(roll_std > 0.02)
    else:
        clear = np.isfinite(tau_app)
    clear &= valid_base[:, None]

    # 正午分界，供半日段回退拟合
    if np.isfinite(am).sum() < 20:
        return {}, pd.DataFrame()
    idx_noon = int(np.nanargmin(np.where(np.isfinite(am), am, np.inf)))
    seg_slices = {'all': slice(0, n),
                  'morning': slice(0, idx_noon + 1),
                  'afternoon': slice(idx_noon, n)}

    # 人工指定定标时段（HH:MM~HH:MM）覆盖分段选择
    if time_range:
        try:
            t_min_arr = np.array([t.hour * 60 + t.minute for t in time_series])
            sh, sm = map(int, str(time_range[0]).split(':'))
            eh, em = map(int, str(time_range[1]).split(':'))
            if sh * 60 + sm < eh * 60 + em:
                ids = np.where((t_min_arr >= sh * 60 + sm) & (t_min_arr <= eh * 60 + em))[0]
                if len(ids) >= min_points:
                    seg_slices = {'manual': slice(int(ids[0]), int(ids[-1]) + 1)}
        except Exception:
            pass

    # 对数信号矩阵 y = ln(E/esd)
    with np.errstate(divide='ignore', invalid='ignore'):
        y_all = np.where(irr > 0, np.log(irr / esd), np.nan)

    def fit_one(x, y):
        """sigma迭代线性回归，返回 (slope, intercept, r2, n_used) 或 None"""
        keep = np.isfinite(y)
        if keep.sum() < min_points:
            return None
        for _ in range(max_iter):
            if keep.sum() < min_points:
                return None
            slope, intercept = np.polyfit(x[keep], y[keep], 1)
            resid = y - (slope * x + intercept)
            std = np.nanstd(resid[keep])
            if std <= 0:
                break
            new_keep = keep & (np.abs(resid) <= sigma * std)
            if (new_keep == keep).all():
                break
            keep = new_keep
        if keep.sum() < min_points:
            return None
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        resid = y[keep] - (slope * x[keep] + intercept)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y[keep] - y[keep].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        # 截距标准误差 σ(lnE0)：低τ通道R²系统性偏低时的估计解判据
        xk = x[keep]
        n_k = int(keep.sum())
        sxx = float(np.sum((xk - xk.mean()) ** 2))
        sigma_e0 = (float(np.sqrt(ss_res / (n_k - 2)) * np.sqrt(1.0 / n_k + float(xk.mean()) ** 2 / sxx))
                    if n_k > 2 and sxx > 0 else np.inf)
        return slope, intercept, r2, n_k, sigma_e0

    # ---- 时段候选：auto 模式为上/下午半日段，日级统一选段 ----
    if 'manual' in seg_slices:
        seg_names = ['manual']
    elif segment in ('morning', 'afternoon', 'all'):
        seg_names = [segment]
    else:  # auto
        seg_names = ['morning', 'afternoon']

    # ---- 第一遍：各波长分别拟合候选时段 ----
    per_wl = []
    for j in range(nwl):
        if progress_cb and j % 50 == 0:
            progress_cb(j / nwl)
        col_clear = clear[:, j]
        y = y_all[:, j]
        res_j = {}
        for seg_name in seg_names:
            sl = seg_slices[seg_name]
            mask = col_clear[sl]
            if mask.sum() < min_points:
                continue
            res = fit_one(am[sl][mask], y[sl][mask])
            if res is None:
                continue
            slope, intercept, r2, n_used, sigma_e0 = res
            # 物理约束优先：斜率负且 τ≤tau_max（排除云崩塌高R²假拟合）；
            # 质量：R²≥min_r2 为正常解；σ(lnE0)≤sigma_e0_max 且 R²≥0.5 为估计解
            phys_ok = (slope < 0) and (-slope <= tau_max)
            ok = bool(phys_ok and (r2 >= min_r2 or (r2 >= 0.5 and sigma_e0 <= sigma_e0_max)))
            res_j[seg_name] = {'e0': float(np.exp(intercept)), 'tau_mean': float(-slope),
                               'r2': float(r2), 'sigma_e0': float(sigma_e0),
                               'n_points': n_used,
                               'estimated': bool(ok and r2 < min_r2), 'success': ok}
        per_wl.append(res_j)

    # ---- 日级统一选段：同一时段应用于全部波长，保持 E0 光谱一致 ----
    # auto 模式上午优先：上午达标波长数不少于下午则统一用上午；
    # 上午整体不达标（如上午阴雨、下午放晴）才统一用下午
    if 'manual' in seg_slices:
        day_segment = 'manual'
    elif segment in ('morning', 'afternoon', 'all'):
        day_segment = segment
    else:  # auto
        n_am = sum(1 for r in per_wl if r.get('morning', {}).get('success'))
        n_pm = sum(1 for r in per_wl if r.get('afternoon', {}).get('success'))
        day_segment = 'morning' if n_am >= n_pm else 'afternoon'

    # ---- 第二遍：按统一时段生成 E0 与报告 ----
    e0_dict = {}
    report_rows = []
    for j, res_j in enumerate(per_wl):
        for seg_name, r in res_j.items():
            report_rows.append({'wavelength': float(wavelengths[j]), 'segment': seg_name,
                                'e0': r['e0'], 'tau_mean': r['tau_mean'],
                                'r2': r['r2'], 'sigma_e0': r['sigma_e0'],
                                'n_points': r['n_points'], 'estimated': r['estimated'],
                                'day_segment': day_segment,
                                'selected': bool(seg_name == day_segment and r['success']),
                                'success': r['success']})
        sel = res_j.get(day_segment)
        if sel is not None and sel['success']:
            e0_dict[float(wavelengths[j])] = sel['e0']

    return e0_dict, pd.DataFrame(report_rows)


# 处理DNI数据，反演936nm水汽光学厚度（870/1020nm基线法）
def process_water_vapor(dni_data, latitude=49.0, longitude=119.4, pressure=840.0,
                        wv_method='standard', e0_langley=None, progress_cb=None,
                        pwv_a=0.585, pwv_b=0.569):
    """
    处理DNI数据，用870/1020nm基线法反演936nm水汽光学厚度 (WVOD)

    流程（对每个时间点）：
    1. 计算太阳天顶角、大气质量数
    2. 由Beer-Lambert定律得870/936/1020nm总光学厚度（不扣气体）
    3. 扣除瑞利光学厚度，得870/1020nm的AOD
    4. 870/1020nm 按幂律内插出936nm非水汽基线
    5. WVOD = τ_total(936) - 基线(936)

    Parameters
    ----------
    dni_data : dict
        DNI数据
    latitude : float
        纬度
    longitude : float
        经度
    pressure : float
        地面气压 (hPa)
    wv_method : str
        基线方法: 'standard'(QX/T 69-2024 公式(8)-(10)，气溶胶+分子散射合计基线)
        或 'aod_only'(纯气溶胶基线，瑞利单独扣除)

    Returns
    -------
    pd.DataFrame
        水汽反演结果，包含 datetime, solar_zenith, aod_870, aod_1020,
        angstrom_alpha, aod_baseline_936, wvod_936 列
    """
    time_series = dni_data['time_series']
    wavelengths = dni_data['wavelengths']
    irradiance_data = dni_data['irradiance']

    # 870/936/1020nm 对应的波长索引
    idx_870 = int(np.argmin(np.abs(wavelengths - 870)))
    idx_936 = int(np.argmin(np.abs(wavelengths - 936)))
    idx_1020 = int(np.argmin(np.abs(wavelengths - 1020)))

    results = []

    total_time_points = len(time_series)
    for i, timestamp in enumerate(time_series):
        # 显示进度
        if i % 10 == 0:
            print(f"水汽反演进度: {i}/{total_time_points} ({i/total_time_points*100:.1f}%)")
            if progress_cb:
                progress_cb(i / total_time_points)

        irradiances = irradiance_data[i, :]

        # 计算太阳天顶角
        try:
            solar_zenith = aod_inversion.solar_zenith_angle(latitude, longitude, timestamp)
        except Exception:
            solar_zenith = 90.0

        if solar_zenith >= 85:
            results.append({
                'datetime': timestamp,
                'solar_zenith': solar_zenith,
                'aod_870': np.nan,
                'aod_1020': np.nan,
                'angstrom_alpha': np.nan,
                'aod_baseline_936': np.nan,
                'wvod_936': np.nan
            })
            continue

        irr_870 = irradiances[idx_870]
        irr_936 = irradiances[idx_936]
        irr_1020 = irradiances[idx_1020]

        if irr_870 <= 0 or irr_936 <= 0 or irr_1020 <= 0:
            results.append({
                'datetime': timestamp,
                'solar_zenith': solar_zenith,
                'aod_870': np.nan,
                'aod_1020': np.nan,
                'angstrom_alpha': np.nan,
                'aod_baseline_936': np.nan,
                'wvod_936': np.nan
            })
            continue

        try:
            # Langley定标的E0（如有）覆盖Wehrli默认值
            e0_ref = None
            if e0_langley:
                e0_ref = dict(aod_inversion.WEHRLI_1985)
                e0_ref.update(e0_langley)
            # 三个波长的总光学厚度与瑞利扣除（气体扣除已屏蔽）
            r870 = aod_inversion.invert_aod(wavelength=870, irradiance=irr_870,
                                            solar_zenith=solar_zenith, date=timestamp,
                                            pressure=pressure, e0_reference=e0_ref)
            r1020 = aod_inversion.invert_aod(wavelength=1020, irradiance=irr_1020,
                                             solar_zenith=solar_zenith, date=timestamp,
                                             pressure=pressure, e0_reference=e0_ref)
            r936 = aod_inversion.invert_aod(wavelength=936, irradiance=irr_936,
                                            solar_zenith=solar_zenith, date=timestamp,
                                            pressure=pressure, e0_reference=e0_ref)

            # 870/1020基线法反演936nm水汽光学厚度
            wv = aod_inversion.invert_wvod_936(
                total_od_936=r936['total_optical_depth'],
                rayleigh_od_936=r936['rayleigh_od'],
                aod_870=r870['aod'],
                aod_1020=r1020['aod'],
                rayleigh_od_870=r870['rayleigh_od'],
                rayleigh_od_1020=r1020['rayleigh_od'],
                method=wv_method
            )

            results.append({
                'datetime': timestamp,
                'solar_zenith': solar_zenith,
                'aod_870': r870['aod'],
                'aod_1020': r1020['aod'],
                'angstrom_alpha': wv['angstrom_alpha'],
                'aod_baseline_936': wv['aod_baseline_936'],
                'wvod_936': wv['wvod']
            })
        except Exception:
            results.append({
                'datetime': timestamp,
                'solar_zenith': solar_zenith,
                'aod_870': np.nan,
                'aod_1020': np.nan,
                'angstrom_alpha': np.nan,
                'aod_baseline_936': np.nan,
                'wvod_936': np.nan
            })

    df = pd.DataFrame(results)
    # 大气质量数（水汽斜程换算与斜柱含量用）
    df['airmass'] = [aod_inversion.calculate_airmass(z) if np.isfinite(z) and z < 90 else np.nan
                     for z in df['solar_zenith'].values]
    # WVOD → 垂直柱可降水量（QX/T 69-2024 式(7)，含 m^b 斜程因子）；斜柱 = m × 垂直柱
    df['pwv_mm'] = aod_inversion.wvod_to_pwv(df['wvod_936'].values, df['airmass'].values,
                                             a=pwv_a, b=pwv_b)
    df['pwv_slant_mm'] = df['pwv_mm'] * df['airmass']
    return df

# ======================= 向量化快速计算（与逐分钟版结果一致） =======================

def _precompute_geometry(time_series, latitude, longitude, airmass_method='kasten'):
    """预计算全天每个时间点的太阳天顶角、大气质量数、日地距离修正因子"""
    n = len(time_series)
    sza = np.full(n, np.nan)
    am = np.full(n, np.nan)
    for i, ts in enumerate(time_series):
        try:
            s = aod_inversion.solar_zenith_angle(latitude, longitude, ts)
            sza[i] = s
            am[i] = aod_inversion.calculate_airmass(s, method=airmass_method)
        except Exception:
            pass
    doy = aod_inversion.calculate_day_of_year(time_series[0])
    esd = aod_inversion.earth_sun_distance_factor(doy)
    return sza, am, esd


def _merge_e0(e0_langley):
    """Langley定标E0（如有）覆盖Wehrli默认值"""
    if e0_langley:
        e0_ref = dict(aod_inversion.WEHRLI_1985)
        e0_ref.update(e0_langley)
        return e0_ref
    return aod_inversion.WEHRLI_1985


def _e0_at(e0_ref, wl):
    if wl in e0_ref:
        return e0_ref[wl]
    waves = np.array(list(e0_ref.keys()))
    vals = np.array(list(e0_ref.values()))
    return float(np.interp(wl, waves, vals))


def _rayleigh_at(wl, pressure, rayleigh_method):
    if rayleigh_method == 'hansen':
        return aod_inversion.rayleigh_optical_depth_hansen(wl, pressure)
    return aod_inversion.rayleigh_optical_depth(wl, pressure)


def _total_od_vec(irr, e0, esd, am):
    """向量化总光学厚度 τ = -ln(E/(E0·esd))/m"""
    total = np.full(len(irr), np.nan)
    valid = np.isfinite(irr) & (irr > 0) & np.isfinite(am) & (am > 0)
    total[valid] = -np.log(irr[valid] / (e0 * esd)) / am[valid]
    return total


def process_dni_data_fast(dni_data, latitude=49.0, longitude=119.4, pressure=840.0,
                          airmass_method='kasten', rayleigh_method='standard',
                          gas_correction=False, e0_langley=None):
    """
    向量化快速版 process_dni_data：太阳几何、瑞利、气体量均按天/波长预计算，
    Beer-Lambert 反演对全天数组一次完成。结果与逐分钟版一致。
    """
    time_series = dni_data['time_series']
    wavelengths = dni_data['wavelengths']
    irr_data = dni_data['irradiance']

    sza, am, esd = _precompute_geometry(time_series, latitude, longitude, airmass_method)
    e0_ref = _merge_e0(e0_langley)
    valid = (sza < 85) & np.isfinite(am)

    key_wavelengths = AOD_KEY_WAVELENGTHS
    aod_cols = {}
    for wl in key_wavelengths:
        idx = int(np.argmin(np.abs(wavelengths - wl)))
        irr = irr_data[:, idx]
        e0 = _e0_at(e0_ref, wl)
        tau_r = _rayleigh_at(wl, pressure, rayleigh_method)
        tau_g = 0.0
        if gas_correction:
            tau_g = aod_inversion.calculate_total_absorption_optical_depth(wl, pressure=pressure)
        total = _total_od_vec(irr, e0, esd, am)
        aod = total - tau_r - tau_g
        aod_cols[wl] = np.where(valid & (irr > 0), np.maximum(aod, 0.0), np.nan)

    # Angstrom 指数（440-870nm），公式与 calculate_angstrom_exponent 一致
    a440, a870 = aod_cols[440], aod_cols[870]
    alpha = np.full(len(time_series), np.nan)
    ok = np.isfinite(a440) & np.isfinite(a870) & (a440 > 0) & (a870 > 0)
    alpha[ok] = -np.log(a440[ok] / a870[ok]) / np.log(440.0 / 870.0)

    df_dict = {'datetime': time_series, 'solar_zenith': sza}
    for wl in key_wavelengths:
        df_dict[f'aod_{wl}'] = aod_cols[wl]
    df_dict['angstrom_alpha'] = alpha
    return pd.DataFrame(df_dict)


def process_water_vapor_fast(dni_data, latitude=49.0, longitude=119.4, pressure=840.0,
                             wv_method='standard', e0_langley=None,
                             pwv_a=0.585, pwv_b=0.569):
    """
    向量化快速版 process_water_vapor（870/1020nm基线法反演936nm水汽光学厚度）
    """
    time_series = dni_data['time_series']
    wavelengths = dni_data['wavelengths']
    irr_data = dni_data['irradiance']

    sza, am, esd = _precompute_geometry(time_series, latitude, longitude)
    e0_ref = _merge_e0(e0_langley)
    valid = (sza < 85) & np.isfinite(am)
    n = len(time_series)

    def col(wl):
        idx = int(np.argmin(np.abs(wavelengths - wl)))
        irr = irr_data[:, idx]
        e0 = _e0_at(e0_ref, wl)
        tau_r = _rayleigh_at(wl, pressure, 'standard')
        total = _total_od_vec(irr, e0, esd, am)
        aod = np.where(valid & (irr > 0), np.maximum(total - tau_r, 0.0), np.nan)
        return irr, total, tau_r, aod

    irr870, total870, r870, aod870 = col(870)
    irr936, total936, r936, _ = col(936)
    irr1020, total1020, r1020, aod1020 = col(1020)

    alpha = np.full(n, np.nan)
    baseline936 = np.full(n, np.nan)
    wvod = np.full(n, np.nan)

    if wv_method == 'aod_only':
        # 纯气溶胶基线
        ok = valid & (aod870 > 0) & (aod1020 > 0) & np.isfinite(total936)
        alpha[ok] = -np.log(aod870[ok] / aod1020[ok]) / np.log(870.0 / 1020.0)
        base = aod870[ok] * (936.0 / 870.0) ** (-alpha[ok])
        baseline936[ok] = base
        wvod[ok] = total936[ok] - r936 - base
    else:
        # QX/T 69-2024 公式(8)-(10)：气溶胶+分子散射合计基线
        tau_am_870 = aod870 + r870
        tau_am_1020 = aod1020 + r1020
        ok = valid & (tau_am_870 > 0) & (tau_am_1020 > 0) & np.isfinite(total936)
        alpha[ok] = -np.log(tau_am_1020[ok] / tau_am_870[ok]) / np.log(1020.0 / 870.0)
        base = tau_am_870[ok] * (936.0 / 870.0) ** (-alpha[ok])
        baseline936[ok] = base
        wvod[ok] = total936[ok] - base

    # WVOD → 垂直柱可降水量（含 m^b 斜程因子）；斜柱 = m × 垂直柱
    pwv = aod_inversion.wvod_to_pwv(wvod, am, a=pwv_a, b=pwv_b)
    return pd.DataFrame({
        'datetime': time_series,
        'solar_zenith': sza,
        'airmass': am,
        'aod_870': aod870,
        'aod_1020': aod1020,
        'angstrom_alpha': alpha,
        'aod_baseline_936': baseline936,
        'wvod_936': wvod,
        'pwv_mm': pwv,
        'pwv_slant_mm': pwv * am
    })


# 绘制波长-AOD曲线
def plot_wavelength_aod(aod_results, time_point=None, time_interval=None):
    """
    绘制波长-AOD曲线
    
    Parameters
    ----------
    aod_results : pd.DataFrame
        AOD反演结果
    time_point : str, optional
        特定时间点，格式为'HH:MM'，例如'12:00'
    time_interval : int, optional
        时间间隔（分钟），用于绘制多个时间点的曲线
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    plt.figure(figsize=(10, 6))
    
    if time_point:
        # 选择特定时间点的数据
        target_time = pd.to_datetime(f'2021-10-31 {time_point}')
        # 找到最接近的时间点
        closest_time = aod_results.iloc[(aod_results['datetime'] - target_time).abs().argsort()[:1]]
        if not closest_time.empty:
            row = closest_time.iloc[0]
            # 波长和对应的AOD值
            wavelengths = [340, 380, 400, 440, 500, 675, 870]
            aod_values = [row['aod_340'], row['aod_380'], row['aod_400'], row['aod_440'], row['aod_500'], row['aod_675'], row['aod_870']]
            
            # 绘制曲线
            plt.plot(wavelengths, aod_values, 'o-', label=f'{row["datetime"].strftime("%H:%M")}')
            plt.title(f'2021年10月31日 {time_point} AOD波长曲线', fontsize=14)
    elif time_interval:
        # 绘制多个时间点的曲线
        # 选择从8:00到16:00的时间点，间隔为time_interval分钟
        start_time = pd.to_datetime('2021-10-31 08:00:00')
        end_time = pd.to_datetime('2021-10-31 16:00:00')
        
        current_time = start_time
        while current_time <= end_time:
            # 找到最接近的时间点
            closest_time = aod_results.iloc[(aod_results['datetime'] - current_time).abs().argsort()[:1]]
            if not closest_time.empty:
                row = closest_time.iloc[0]
                # 波长和对应的AOD值
                wavelengths = [340, 380, 400, 440, 500, 675, 870]
                aod_values = [row['aod_340'], row['aod_380'], row['aod_400'], row['aod_440'], row['aod_500'], row['aod_675'], row['aod_870']]
                
                # 绘制曲线
                plt.plot(wavelengths, aod_values, 'o-', label=f'{row["datetime"].strftime("%H:%M")}')
            current_time += pd.Timedelta(minutes=time_interval)
        plt.title(f'2021年10月31日AOD波长曲线 (间隔{time_interval}分钟)', fontsize=14)
    else:
        # 默认绘制正午12点的数据
        target_time = pd.to_datetime('2021-10-31 12:00:00')
        closest_time = aod_results.iloc[(aod_results['datetime'] - target_time).abs().argsort()[:1]]
        if not closest_time.empty:
            row = closest_time.iloc[0]
            # 波长和对应的AOD值
            wavelengths = [340, 380, 400, 440, 500, 675, 870]
            aod_values = [row['aod_340'], row['aod_380'], row['aod_400'], row['aod_440'], row['aod_500'], row['aod_675'], row['aod_870']]
            
            # 绘制曲线
            plt.plot(wavelengths, aod_values, 'o-', label=f'{row["datetime"].strftime("%H:%M")}')
            plt.title('2021年10月31日正午AOD波长曲线', fontsize=14)
    
    # 设置图表属性
    plt.xlabel('波长 (nm)', fontsize=12)
    plt.ylabel('AOD', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # 保存图表
    plt.tight_layout()
    if time_point:
        filename = f'20211031_aod_wavelength_{time_point.replace(":", "")}.png'
    elif time_interval:
        filename = f'20211031_aod_wavelength_interval{time_interval}.png'
    else:
        filename = '20211031_aod_wavelength_noon.png'
    plt.savefig(filename, dpi=300)
    print(f"AOD波长曲线已保存: {filename}")
    
    # 关闭图表，避免阻塞
    plt.close()

# 绘制高光谱光学厚度曲线
def plot_hyperspectral_optical_depth(hyperspectral_data):
    """
    绘制高光谱光学厚度曲线
    
    Parameters
    ----------
    hyperspectral_data : dict
        包含不同时刻高光谱光学厚度数据的字典
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    plt.figure(figsize=(12, 8))
    
    # 定义不同时刻的颜色
    colors = {
        '08:00': 'blue',
        '10:00': 'green',
        '12:00': 'red',
        '14:00': 'purple',
        '16:00': 'orange'
    }
    
    # 绘制每个时刻的光学厚度曲线
    for time_str, data in hyperspectral_data.items():
        wavelengths = data['wavelengths']
        optical_depths = data['optical_depths']
        solar_zenith = data['solar_zenith']
        
        # 过滤掉NaN值
        valid_idx = np.where(~np.isnan(optical_depths))[0]
        valid_wavelengths = np.array(wavelengths)[valid_idx]
        valid_od = np.array(optical_depths)[valid_idx]
        
        # 绘制曲线
        color = colors.get(time_str, 'gray')
        plt.plot(valid_wavelengths, valid_od, '-', label=f'{time_str} (SZA: {solar_zenith:.1f}°)', color=color, linewidth=1.5)
        
        # 标记特定波长点（340, 380, 400, 500, 675, 870, 1020nm）
        target_wavelengths = [340, 380, 400, 500, 675, 870, 1020]
        for target_wave in target_wavelengths:
            # 找到最接近的波长索引
            idx = np.argmin(np.abs(np.array(wavelengths) - target_wave))
            if not np.isnan(optical_depths[idx]):
                plt.scatter(wavelengths[idx], optical_depths[idx], color='yellow', edgecolor='black', s=50, zorder=5)
    
    # 设置图表属性
    plt.title('2021年10月31日高光谱光学厚度曲线', fontsize=16)
    plt.xlabel('Wavelength (nm)', fontsize=14)
    plt.ylabel('Aerosols and gass optical thickness (a.u.)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    
    # 设置坐标轴范围
    plt.xlim(300, 1100)
    plt.ylim(0, None)  # 自动调整纵轴上限
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('20211031_hyperspectral_optical_depth.png', dpi=300)
    print("高光谱光学厚度曲线已保存: 20211031_hyperspectral_optical_depth.png")
    
    # 关闭图表
    plt.close()

# 主函数
if __name__ == "__main__":
    # 读取数据
    print("读取DNI数据...")
    dni_data = read_dni_data('20211031svd.xlsx')
    print(f"数据读取完成: {len(dni_data['time_series'])}个时间点, {len(dni_data['wavelengths'])}个波长")
    
    # 处理数据
    print("处理DNI数据，反演AOD...")
    aod_results = process_dni_data(dni_data)
    print("AOD反演完成")
    
    # 保存结果
    output_file = '20211031_aod_results.xlsx'
    aod_results.to_excel(output_file, index=False)
    print(f"分钟AOD结果保存到: {output_file}")
    
    # 计算日平均AOD
    daily_avg = calculate_daily_average(aod_results)
    print("\n日平均AOD结果:")
    for key, value in daily_avg.items():
        print(f"{key}: {value:.4f}")
    
    # 保存日平均结果
    daily_avg_df = pd.DataFrame([daily_avg])
    daily_avg_df.to_excel('20211031_daily_average_aod.xlsx', index=False)
    print("\n日平均AOD结果保存到: 20211031_daily_average_aod.xlsx")
    
    # 绘制AOD时间序列图
    print("\n绘制AOD时间序列图...")
    plot_aod_time_series(aod_results)
    
    # 绘制正午12点的波长-AOD曲线
    print("\n绘制正午12点的波长-AOD曲线...")
    plot_wavelength_aod(aod_results, time_point='12:00')
    
    # 绘制每2小时的波长-AOD曲线
    print("\n绘制每2小时的波长-AOD曲线...")
    plot_wavelength_aod(aod_results, time_interval=120)
    
    # 处理高光谱数据并绘制曲线
    print("\n处理高光谱数据...")
    hyperspectral_data = process_hyperspectral_data(dni_data)
    print("绘制高光谱光学厚度曲线...")
    plot_hyperspectral_optical_depth(hyperspectral_data)
    
    # 显示前几行结果
    print("\n反演结果预览:")
    print(aod_results.head())