# -*- coding: utf-8 -*-
import os
import json
import io
import base64
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
if os.getcwd() != script_dir:
    os.chdir(script_dir)
    print(f"Working directory changed to: {script_dir}")

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import pandas as pd
import numpy as np
import threading
import uuid
from datetime import datetime, timedelta
import aod_inversion
import process_dni_data

result_cache = {}
task_store = {}  # 后台计算任务: task_id -> {status, progress, stage, response/message}

def clean_for_json(obj):
    """Clean NaN and infinity values for JSON serialization"""
    try:
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return clean_for_json(obj.tolist())
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    except (TypeError, ValueError):
        return None

# 软件版本与信息（About 对话框与 /api/about 的单一来源，改版时只改这里）
APP_VERSION = "1.0.0"
APP_INFO = {
    "name": "高光谱AOD/WVOD反演软件 (Hyperspectral AOD/WVOD Inversion System)",
    "version": APP_VERSION,
    "authors": ["Liu Liying", "Zheng Feng"],
    "contact": "feng1214@126.com",
    "copyright": "Copyright © 2026 Liu Liying, Zheng Feng",
    "standard": "QX/T 69-2024 气溶胶光学厚度 太阳光度计法"
}

# Create FastAPI app with custom Swagger UI configuration
app = FastAPI(
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_parameters={
        "dom_id": "#swagger-ui",
        "layout": "BaseLayout",
        "deepLinking": True,
        "showExtensions": True,
        "showCommonExtensions": True,
        "defaultModelsExpandDepth": 2,
        "defaultModelExpandDepth": 2,
        "defaultModelRendering": "model",
        "displayRequestDuration": True,
        "docExpansion": "none",
        "filter": False,
        "operationsSorter": "alpha",
        "tagsSorter": "alpha",
        "tryItOutEnabled": True,
        "validatorUrl": None,
        "syntaxHighlight": {
            "activate": True,
            "theme": "agate"
        }
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=script_dir), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(script_dir, "frontend_english.html"))

@app.get("/api/about")
async def api_about():
    """软件版本与版权信息（前端 About 对话框数据源）"""
    return APP_INFO

@app.post("/aod-inversion")
async def aod_inversion_endpoint(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    pressure: float = Form(840.0),
    ozone_du: float = Form(...),
    airmass_method: str = Form('kasten'),
    rayleigh_method: str = Form('standard'),
    gas_correction: bool = Form(False),
    langley_cal: bool = Form(True),
    calc_mode: str = Form('fast'),
    exclude_periods: str = Form('[]'),
    pressure_mode: str = Form('altitude'),
    altitude: str = Form(''),
    langley_segment: str = Form('auto'),
    langley_start: str = Form(''),
    langley_end: str = Form(''),
    data_start_time: str = Form(''),
    data_end_time: str = Form(''),
    data_wl_start: str = Form(''),
    data_wl_end: str = Form('')
):
    """提交AOD反演后台任务，立即返回task_id，前端轮询 /task-status/{task_id} 获取进度"""
    contents = await file.read()
    # 气压来源：由海拔计算（标准大气压高公式）或直接使用输入气压
    if pressure_mode == 'altitude':
        try:
            pressure = aod_inversion.pressure_from_altitude(float(altitude))
            print(f"海拔 {altitude} m → 气压 {pressure:.1f} hPa")
        except (ValueError, TypeError):
            return JSONResponse(status_code=400, content={"detail": "海拔无效，无法计算气压 / Invalid altitude"})
    # 人工定标时段：仅当分段选择为 manual 时生效（覆盖 auto/上午/下午/全天）
    time_range = None
    if langley_segment == 'manual':
        try:
            sh, sm = map(int, langley_start.strip().split(':'))
            eh, em = map(int, langley_end.strip().split(':'))
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "选择人工定标时段时请用时分控件填写起止时间 / Invalid time format"})
        if (sh, sm) >= (eh, em):
            return JSONResponse(status_code=400, content={"detail": "定标时段起始需早于结束 / Start must be before end"})
        time_range = (langley_start.strip(), langley_end.strip())
    task_id = uuid.uuid4().hex[:12]
    task_store[task_id] = {'status': 'running', 'progress': 0.0, 'stage': 'Queued'}
    # 数据格式覆盖参数（时段/波长范围，空=默认 04:00 起1分钟、300-1100nm）
    data_meta = {'data_start': data_start_time, 'data_end': data_end_time,
                 'wl_start': data_wl_start, 'wl_end': data_wl_end}
    threading.Thread(
        target=_run_aod_task,
        args=(task_id, contents, file.filename, latitude, longitude, pressure,
              ozone_du, airmass_method, rayleigh_method, gas_correction,
              langley_cal, calc_mode, exclude_periods, langley_segment, time_range,
              data_meta),
        daemon=True
    ).start()
    return {"status": "started", "task_id": task_id}


def _build_raw_view(dni_data, langley_plots):
    """构建原始数据查看：整点光谱、特征波长辐照度时间序列、Langley拟合图数据"""
    times = dni_data['time_series']
    wl = np.asarray(dni_data['wavelengths'], dtype=float)
    irr = dni_data['irradiance']
    view = {'wavelengths': [round(float(x), 1) for x in wl],
            'time': [t.strftime('%H:%M') for t in times]}
    # 1) 整点光谱：每个整点取最接近该整点的记录
    hourly = []
    minutes = np.array([t.hour * 60 + t.minute for t in times])
    for h in sorted(set(t.hour for t in times)):
        j = int(np.argmin(np.abs(minutes - h * 60)))
        hourly.append({'label': f"{h:02d}:00",
                       'values': [None if not np.isfinite(v) else float(v) for v in irr[j]]})
    view['hourly_spectra'] = hourly
    # 2) 特征波长辐照度时间序列
    irr_ts = {}
    for k in [340, 380, 400, 440, 500, 675, 870, 936, 1020]:
        idx = int(np.argmin(np.abs(wl - k)))
        irr_ts[str(k)] = [None if not np.isfinite(v) else float(v) for v in irr[:, idx]]
    view['irr_timeseries'] = irr_ts
    # 3) Langley 拟合图数据（langley_cal=False 时为空）
    view['langley'] = langley_plots or {}
    return view


def _run_aod_task(task_id, contents, filename, latitude, longitude, pressure,
                  ozone_du, airmass_method, rayleigh_method, gas_correction,
                  langley_cal, calc_mode, exclude_periods='[]', langley_segment='auto',
                  time_range=None, data_meta=None):
    """后台执行AOD反演，进度写入 task_store"""
    def set_prog(p, stage):
        task_store[task_id].update(progress=round(min(p, 99.0), 1), stage=stage)
    try:
        set_prog(2, 'Reading data file')
        dni_data = process_dni_data.read_dni_data(io.BytesIO(contents), original_filename=filename,
                                                  **(data_meta or {}))
        # 剔除时段：整行置NaN，不参与Langley定标与反演
        excluded = process_dni_data.apply_exclude_periods(dni_data, json.loads(exclude_periods or '[]'))
        # Langley定标（默认开启，失败波长自动回退Wehrli标准值）
        e0_langley = None
        e0_spectrum = None
        langley_plots = None
        if langley_cal:
            e0_langley, _, langley_plots = process_dni_data.langley_calibrate_day(
                dni_data, latitude=latitude, longitude=longitude, return_plot=True,
                segment=langley_segment, time_range=time_range,
                progress_cb=lambda f: set_prog(5 + f * 10, 'Langley calibration (key wavelengths)'))
            # 全波段Langley定标（用于高光谱AOD曲线）
            e0_spectrum, _ = process_dni_data.langley_calibrate_spectrum(
                dni_data, latitude=latitude, longitude=longitude,
                airmass_method=airmass_method,
                segment=langley_segment, time_range=time_range,
                progress_cb=lambda f: set_prog(15 + f * 15, 'Langley calibration (full spectrum)'))
        # Process DNI data, invert AOD
        # calc_mode: 'fast'(向量化) / 'slow'(逐分钟) / 'both'(两者都算，fast结果以_fast后缀返回)
        aod_fast = None
        if calc_mode in ('fast', 'both'):
            set_prog(30, 'AOD inversion (fast)')
            aod_fast = process_dni_data.process_dni_data_fast(
                dni_data, latitude=latitude, longitude=longitude, pressure=pressure,
                airmass_method=airmass_method, rayleigh_method=rayleigh_method,
                gas_correction=gas_correction, e0_langley=e0_langley)
        aod_results = None
        if calc_mode in ('slow', 'both'):
            aod_results = process_dni_data.process_dni_data(
                dni_data, latitude=latitude, longitude=longitude, pressure=pressure,
                airmass_method=airmass_method, rayleigh_method=rayleigh_method,
                gas_correction=gas_correction, e0_langley=e0_langley,
                progress_cb=lambda f: set_prog(30 + f * 40, 'AOD inversion (per-minute)'))
        if aod_results is None:
            aod_results = aod_fast
        # Process hyperspectral data（向量化，全波段Langley定标E0）
        hyperspectral_data = process_dni_data.process_hyperspectral_data(
            dni_data, latitude=latitude, longitude=longitude, pressure=pressure, ozone_du=ozone_du,
            airmass_method=airmass_method, rayleigh_method=rayleigh_method,
            gas_correction=gas_correction, e0_langley=e0_spectrum,
            progress_cb=lambda f: set_prog(70 + f * 28, 'Hyperspectral processing'))

        # Prepare result
        # Get wavelengths from the first time point (all time points should have the same wavelengths)
        wavelengths = []
        if hyperspectral_data:
            first_key = list(hyperspectral_data.keys())[0]
            wavelengths = hyperspectral_data[first_key].get('wavelengths', []).tolist()

        # 特征波长的AOD（与 process_dni_data.AOD_KEY_WAVELENGTHS 保持一致）
        aod_wls = process_dni_data.AOD_KEY_WAVELENGTHS
        result = {
            "time": [t.strftime('%H:%M') for t in aod_results['datetime']],
            "data_date": aod_results['datetime'].iloc[0].strftime('%Y-%m-%d'),
            "times": list(hyperspectral_data.keys()),
            "wavelengths": wavelengths,
            "wavelength_series": [data['aod'] for data in hyperspectral_data.values()],
            "wavelength_series_total": [data['optical_depths'] for data in hyperspectral_data.values()],
            "excluded_periods": excluded,
            "raw_view": _build_raw_view(dni_data, langley_plots)
        }
        for wl in aod_wls:
            col = f'aod_{wl}'
            if col in aod_results.columns:
                result[col] = aod_results[col].tolist()
        # both 模式：附加向量化快速版结果供对比
        if calc_mode == 'both' and aod_fast is not None:
            for wl in aod_wls:
                col = f'aod_{wl}'
                if col in aod_fast.columns:
                    result[f'{col}_fast'] = aod_fast[col].tolist()

        # Clean result for JSON serialization (handle NaN values)
        result = clean_for_json(result)

        result_id = f"aod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result_cache[result_id] = result
        task_store[task_id].update(
            status='done', progress=100.0, stage='Done',
            response={
                "status": "success",
                "message": "AOD inversion completed",
                "result_id": result_id,
                "result": result
            })
    except Exception as e:
        task_store[task_id].update(status='error', message=str(e))

@app.post("/process-dni")
async def process_dni_endpoint(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    pressure: float = Form(...)
):
    try:
        contents = await file.read()
        file_data = io.BytesIO(contents)
        # Read DNI data
        dni_data = process_dni_data.read_dni_data(file_data, original_filename=file.filename)
        # Invert ozone
        ozone_results = process_dni_data.invert_ozone(dni_data, latitude=latitude, longitude=longitude, pressure=pressure)
        
        # Prepare result
        result = {
            "time": [t.strftime('%H:%M') for t in ozone_results['datetime']],
            "ozone_avg": ozone_results['average_ozone'].tolist(),
            "ozone_305": ozone_results['ozone_305'].tolist(),
            "ozone_310": ozone_results['ozone_310'].tolist(),
            "ozone_320": ozone_results['ozone_320'].tolist(),
            "ozone_340": ozone_results['ozone_340'].tolist(),
            "ozone_360": ozone_results['ozone_360'].tolist(),
            "ozone_380": ozone_results['ozone_380'].tolist()
        }
        
        # Clean result for JSON serialization (handle NaN values)
        result = clean_for_json(result)
        
        result_id = f"dni_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result_cache[result_id] = result
        response = {
            "status": "success",
            "message": "DNI processing completed",
            "result_id": result_id,
            "result": result
        }
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        error_message = str(e)
        if "nan" in error_message.lower() or "NaN" in error_message:
            error_message = "Processing failed: Invalid data values (NaN) detected in input data"
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": error_message}
        )

@app.post("/water-vapor")
async def water_vapor_endpoint(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    pressure: float = Form(840.0),
    wv_method: str = Form('standard'),
    langley_cal: bool = Form(True),
    calc_mode: str = Form('fast'),
    pwv_a: float = Form(0.585),
    pwv_b: float = Form(0.569),
    exclude_periods: str = Form('[]'),
    pressure_mode: str = Form('altitude'),
    altitude: str = Form(''),
    langley_segment: str = Form('auto'),
    langley_start: str = Form(''),
    langley_end: str = Form(''),
    data_start_time: str = Form(''),
    data_end_time: str = Form(''),
    data_wl_start: str = Form(''),
    data_wl_end: str = Form('')
):
    """提交水汽反演后台任务，立即返回task_id，前端轮询 /task-status/{task_id} 获取进度"""
    contents = await file.read()
    # 气压来源：由海拔计算（标准大气压高公式）或直接使用输入气压
    if pressure_mode == 'altitude':
        try:
            pressure = aod_inversion.pressure_from_altitude(float(altitude))
            print(f"海拔 {altitude} m → 气压 {pressure:.1f} hPa")
        except (ValueError, TypeError):
            return JSONResponse(status_code=400, content={"detail": "海拔无效，无法计算气压 / Invalid altitude"})
    # 人工定标时段：仅当分段选择为 manual 时生效（覆盖 auto/上午/下午/全天）
    time_range = None
    if langley_segment == 'manual':
        try:
            sh, sm = map(int, langley_start.strip().split(':'))
            eh, em = map(int, langley_end.strip().split(':'))
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "选择人工定标时段时请用时分控件填写起止时间 / Invalid time format"})
        if (sh, sm) >= (eh, em):
            return JSONResponse(status_code=400, content={"detail": "定标时段起始需早于结束 / Start must be before end"})
        time_range = (langley_start.strip(), langley_end.strip())
    task_id = uuid.uuid4().hex[:12]
    task_store[task_id] = {'status': 'running', 'progress': 0.0, 'stage': 'Queued'}
    data_meta = {'data_start': data_start_time, 'data_end': data_end_time,
                 'wl_start': data_wl_start, 'wl_end': data_wl_end}
    threading.Thread(
        target=_run_wv_task,
        args=(task_id, contents, file.filename, latitude, longitude, pressure,
              wv_method, langley_cal, calc_mode, pwv_a, pwv_b, exclude_periods,
              langley_segment, time_range, data_meta),
        daemon=True
    ).start()
    return {"status": "started", "task_id": task_id}


def _run_wv_task(task_id, contents, filename, latitude, longitude, pressure,
                 wv_method, langley_cal, calc_mode, pwv_a=0.585, pwv_b=0.569,
                 exclude_periods='[]', langley_segment='auto', time_range=None,
                 data_meta=None):
    """后台执行水汽反演，进度写入 task_store"""
    def set_prog(p, stage):
        task_store[task_id].update(progress=round(min(p, 99.0), 1), stage=stage)
    try:
        set_prog(2, 'Reading data file')
        dni_data = process_dni_data.read_dni_data(io.BytesIO(contents), original_filename=filename,
                                                  **(data_meta or {}))
        # 剔除时段：整行置NaN，不参与Langley定标与反演
        excluded = process_dni_data.apply_exclude_periods(dni_data, json.loads(exclude_periods or '[]'))
        # Langley定标（默认开启，失败波长自动回退Wehrli标准值）
        e0_langley = None
        langley_plots = None
        if langley_cal:
            e0_langley, _, langley_plots = process_dni_data.langley_calibrate_day(
                dni_data, latitude=latitude, longitude=longitude, return_plot=True,
                segment=langley_segment, time_range=time_range,
                progress_cb=lambda f: set_prog(5 + f * 25, 'Langley calibration'))
        # Invert water vapor optical depth at 936nm (870/1020nm baseline method)
        # calc_mode: 'fast'(向量化) / 'slow'(逐分钟) / 'both'(两者都算，fast结果以_fast后缀返回)
        wv_fast = None
        if calc_mode in ('fast', 'both'):
            set_prog(35, 'Water vapor inversion (fast)')
            wv_fast = process_dni_data.process_water_vapor_fast(
                dni_data, latitude=latitude, longitude=longitude, pressure=pressure,
                wv_method=wv_method, e0_langley=e0_langley, pwv_a=pwv_a, pwv_b=pwv_b)
        wv_results = None
        if calc_mode in ('slow', 'both'):
            wv_results = process_dni_data.process_water_vapor(
                dni_data, latitude=latitude, longitude=longitude, pressure=pressure,
                wv_method=wv_method, e0_langley=e0_langley, pwv_a=pwv_a, pwv_b=pwv_b,
                progress_cb=lambda f: set_prog(35 + f * 60, 'Water vapor inversion (per-minute)'))
        if wv_results is None:
            wv_results = wv_fast

        # Prepare result
        result = {
            "time": [t.strftime('%H:%M') for t in wv_results['datetime']],
            "data_date": wv_results['datetime'].iloc[0].strftime('%Y-%m-%d'),
            "wvod_936": wv_results['wvod_936'].tolist(),
            "pwv_mm": wv_results['pwv_mm'].tolist(),
            "pwv_slant_mm": wv_results['pwv_slant_mm'].tolist(),
            "aod_870": wv_results['aod_870'].tolist(),
            "aod_1020": wv_results['aod_1020'].tolist(),
            "angstrom_alpha": wv_results['angstrom_alpha'].tolist(),
            "aod_baseline_936": wv_results['aod_baseline_936'].tolist(),
            "pwv_a": pwv_a,
            "pwv_b": pwv_b,
            "excluded_periods": excluded,
            "raw_view": _build_raw_view(dni_data, langley_plots)
        }
        # both 模式：附加向量化快速版结果供对比
        if calc_mode == 'both' and wv_fast is not None:
            result["wvod_936_fast"] = wv_fast['wvod_936'].tolist()
            result["pwv_mm_fast"] = wv_fast['pwv_mm'].tolist()
            result["pwv_slant_mm_fast"] = wv_fast['pwv_slant_mm'].tolist()
            result["aod_870_fast"] = wv_fast['aod_870'].tolist()
            result["aod_1020_fast"] = wv_fast['aod_1020'].tolist()

        # Clean result for JSON serialization (handle NaN values)
        result = clean_for_json(result)

        result_id = f"wv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result_cache[result_id] = result
        task_store[task_id].update(
            status='done', progress=100.0, stage='Done',
            response={
                "status": "success",
                "message": "Water vapor inversion completed",
                "result_id": result_id,
                "result": result
            })
    except Exception as e:
        task_store[task_id].update(status='error', message=str(e))


@app.get("/task-status/{task_id}")
async def task_status_endpoint(task_id: str):
    """查询后台计算任务进度"""
    task = task_store.get(task_id)
    if task is None:
        return JSONResponse(status_code=404, content={"status": "error", "message": "Task not found"})
    return task


def _build_svg_view(svg_data):
    """构建svg查看数据：整点光谱（每个整点取最接近的记录）"""
    times = svg_data['time_series']
    irr = svg_data['irradiance']
    hourly = []
    minutes = np.array([t.hour * 60 + t.minute for t in times])
    for h in sorted(set(t.hour for t in times)):
        j = int(np.argmin(np.abs(minutes - h * 60)))
        hourly.append({'label': f"{h:02d}:00",
                       'values': [None if not np.isfinite(v) else float(v) for v in irr[j]]})
    return {'hourly_spectra': hourly}


def _run_svg_task(task_id, contents, filename, exclude_periods='[]', data_meta=None):
    """后台处理svg：读取文件→积分计算→视图数据，进度写入 task_store"""
    def set_prog(p, stage):
        task_store[task_id].update(progress=round(min(p, 99.0), 1), stage=stage)
    try:
        set_prog(5, 'Reading data file')
        svg_data = process_dni_data.read_svg_data(io.BytesIO(contents), original_filename=filename,
                                                  **(data_meta or {}))
        # 剔除时段：整行置NaN，不参与积分与绘图
        excluded = process_dni_data.apply_exclude_periods(svg_data, json.loads(exclude_periods or '[]'))
        set_prog(60, 'Computing spectral integrals')
        integ = process_dni_data.compute_svg_integrals(svg_data['wavelengths'], svg_data['irradiance'])
        set_prog(85, 'Building view data')
        view = _build_svg_view(svg_data)
        result = {
            "time": [t.strftime('%H:%M') for t in svg_data['time_series']],
            "data_date": svg_data['time_series'][0].strftime('%Y-%m-%d'),
            "wavelengths": [round(float(x), 1) for x in svg_data['wavelengths']],
            "integrals": {k: v.tolist() for k, v in integ.items()},
            "excluded_periods": excluded,
            "hourly_spectra": view['hourly_spectra']
        }
        result = clean_for_json(result)
        result_id = f"svg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result_cache[result_id] = result
        task_store[task_id].update(
            status='done', progress=100.0, stage='Done',
            response={"status": "success", "message": "SVG data loaded",
                      "result_id": result_id, "result": result})
    except Exception as e:
        task_store[task_id].update(status='error', message=str(e))


@app.post("/view-svg")
async def view_svg_endpoint(
    file: UploadFile = File(...),
    exclude_periods: str = Form('[]'),
    data_start_time: str = Form(''),
    data_end_time: str = Form(''),
    data_wl_start: str = Form(''),
    data_wl_end: str = Form('')
):
    """提交svg查看后台任务（与反演一致的异步模式，前端轮询 /task-status 获取进度）"""
    contents = await file.read()
    task_id = uuid.uuid4().hex[:12]
    task_store[task_id] = {'status': 'running', 'progress': 0.0, 'stage': 'Queued'}
    # 数据格式覆盖参数（时段/波长范围，空=默认 04:00 起1分钟、按列数识别波段）
    data_meta = {'data_start': data_start_time, 'data_end': data_end_time,
                 'wl_start': data_wl_start, 'wl_end': data_wl_end}
    threading.Thread(target=_run_svg_task,
                     args=(task_id, contents, file.filename, exclude_periods, data_meta),
                     daemon=True).start()
    return {"status": "started", "task_id": task_id}


@app.post("/save-svg-results")
async def save_svg_results_endpoint(result_id: str = Form(...), filename: str = Form(...)):
    """导出svg积分量与整点光谱到 Excel"""
    try:
        if result_id not in result_cache:
            return JSONResponse(status_code=404, content={"status": "error", "message": "Result not found"})
        result = result_cache[result_id]

        svg_results_dir = os.path.join(script_dir, "SVG results")
        os.makedirs(svg_results_dir, exist_ok=True)
        date_dir = os.path.join(svg_results_dir, datetime.now().strftime("%Y%m%d"))
        os.makedirs(date_dir, exist_ok=True)
        file_results_dir = os.path.join(date_dir, f"{filename} results")
        os.makedirs(file_results_dir, exist_ok=True)

        excel_path = os.path.join(file_results_dir, f"{filename}_svg_integrals.xlsx")
        integ = result.get('integrals', {})
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df = pd.DataFrame({
                "Time": result.get("time", []),
                "Total (W/m2)": integ.get('total'),
                "UV 300-400nm (W/m2)": integ.get('uv'),
                "VIS 400-700nm (W/m2)": integ.get('vis'),
                "NIR 700-1100nm (W/m2)": integ.get('nir'),
                "PAR 400-700nm (W/m2)": integ.get('par_w'),
                "PAR PPFD (umol/m2/s)": integ.get('par_ppfd'),
                "Illuminance (lux)": integ.get('lux'),
            })
            df.to_excel(writer, sheet_name="Integrals", index=False)
            # 整点光谱：行=波长，列=整点时刻
            hourly = result.get('hourly_spectra') or []
            if hourly:
                spec = {"Wavelength (nm)": result.get("wavelengths", [])}
                for h in hourly:
                    spec[h['label']] = h['values']
                pd.DataFrame(spec).to_excel(writer, sheet_name="Hourly Spectra", index=False)

        return {"status": "success", "message": "SVG results saved successfully", "excel_path": excel_path}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Failed to save svg results: {e}"})

@app.post("/save-results")
async def save_results_endpoint(
    result_id: str = Form(...),
    filename: str = Form(...)
):
    try:
        # Get result from cache
        if result_id not in result_cache:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Result not found"}
            )
        
        result = result_cache[result_id]
        
        # Create AOD results directory if it doesn't exist
        aod_results_dir = os.path.join(script_dir, "AOD results")
        if not os.path.exists(aod_results_dir):
            os.makedirs(aod_results_dir)
        
        # Create subdirectory for the current date if it doesn't exist
        date_dir = os.path.join(aod_results_dir, datetime.now().strftime("%Y%m%d"))
        if not os.path.exists(date_dir):
            os.makedirs(date_dir)
        
        # Create subdirectory for the specific file results
        file_results_dir = os.path.join(date_dir, f"{filename} results")
        if not os.path.exists(file_results_dir):
            os.makedirs(file_results_dir)
        
        # Save results to JSON file
        json_filename = f"result_{time.time()}.json"
        json_path = os.path.join(file_results_dir, json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Save results to Excel file
        excel_filename = f"{filename}_results.xlsx"
        excel_path = os.path.join(file_results_dir, excel_filename)
        
        # Create Excel writer
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            # Save time series data (导出全部可用的AOD波长)
            if "time" in result and "aod_440" in result:
                time_series_data = {"Time": result["time"]}
                for wl in process_dni_data.AOD_KEY_WAVELENGTHS:
                    col = f"aod_{wl}"
                    if col in result:
                        time_series_data[f"AOD {wl}nm"] = result[col]
                time_series_df = pd.DataFrame(time_series_data)
                time_series_df.to_excel(writer, sheet_name="Time Series", index=False)

            # Save water vapor results
            if "time" in result and "wvod_936" in result:
                wv_data = {
                    "Time": result["time"],
                    "WVOD 936nm": result["wvod_936"],
                    "PWV vertical (mm)": result.get("pwv_mm"),
                    "PWV slant (mm)": result.get("pwv_slant_mm"),
                    "AOD 870nm": result["aod_870"],
                    "AOD 1020nm": result["aod_1020"],
                    "Angstrom Alpha": result["angstrom_alpha"],
                    "Baseline 936nm": result["aod_baseline_936"]
                }
                wv_df = pd.DataFrame(wv_data)
                wv_df.to_excel(writer, sheet_name="Water Vapor", index=False)
            
            # Save hyperspectral data
            if "wavelengths" in result and "wavelength_series" in result and "times" in result:
                for i, (time_str, series) in enumerate(zip(result["times"], result["wavelength_series"])):
                    if series:
                        hyperspectral_data = {
                            "Wavelength (nm)": result["wavelengths"],
                            "Optical Depth": series
                        }
                        hyperspectral_df = pd.DataFrame(hyperspectral_data)
                        # Replace colons in sheet name because Excel doesn't allow colons in sheet titles
                        sheet_name = f"{time_str}".replace(':', '-')
                        if len(sheet_name) > 31:  # Excel sheet name limit
                            sheet_name = sheet_name[:31]
                        hyperspectral_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        response = {
            "status": "success",
            "message": "Results saved successfully",
            "json_path": json_path,
            "excel_path": excel_path
        }
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        error_message = str(e)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to save results: {error_message}"}
        )

@app.post("/save-charts")
async def save_charts_endpoint(
    result_id: str = Form(...),
    filename: str = Form(...),
    time_series_chart: str = Form(None),
    wavelength_chart: str = Form(None),
    spectrum_chart: str = Form(None),
    integral_chart: str = Form(None),
    ppfd_lux_chart: str = Form(None)
):
    try:
        # Get result from cache
        if result_id not in result_cache:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Result not found"}
            )
        
        # Create AOD results directory if it doesn't exist
        aod_results_dir = os.path.join(script_dir, "AOD results")
        if not os.path.exists(aod_results_dir):
            os.makedirs(aod_results_dir)
        
        # Create subdirectory for the current date if it doesn't exist
        date_dir = os.path.join(aod_results_dir, datetime.now().strftime("%Y%m%d"))
        if not os.path.exists(date_dir):
            os.makedirs(date_dir)
        
        # Create subdirectory for the specific file results
        file_results_dir = os.path.join(date_dir, f"{filename} results")
        if not os.path.exists(file_results_dir):
            os.makedirs(file_results_dir)
        
        # Create charts directory
        charts_dir = os.path.join(file_results_dir, f"{filename}_charts")
        if not os.path.exists(charts_dir):
            os.makedirs(charts_dir)
        
        # Save time series chart
        if time_series_chart:
            time_series_path = os.path.join(charts_dir, "Time Series.png")
            with open(time_series_path, "wb") as f:
                f.write(base64.b64decode(time_series_chart))
        
        # Save wavelength chart
        if wavelength_chart:
            wavelength_path = os.path.join(charts_dir, "Wavelength Curve.png")
            with open(wavelength_path, "wb") as f:
                f.write(base64.b64decode(wavelength_chart))

        # Save svg charts（水平总辐射查看）
        for img, name in ((spectrum_chart, "Hourly Spectra.png"),
                          (integral_chart, "Integral Time Series.png"),
                          (ppfd_lux_chart, "PAR and Illuminance.png")):
            if img:
                with open(os.path.join(charts_dir, name), "wb") as f:
                    f.write(base64.b64decode(img))
        
        response = {
            "status": "success",
            "message": "Charts saved successfully",
            "charts_dir": charts_dir
        }
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        error_message = str(e)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to save charts: {error_message}"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)