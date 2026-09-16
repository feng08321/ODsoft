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

# Create FastAPI app with custom Swagger UI configuration
app = FastAPI(
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

@app.post("/aod-inversion")
async def aod_inversion_endpoint(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    pressure: float = Form(...),
    ozone_du: float = Form(...),
    airmass_method: str = Form('kasten'),
    rayleigh_method: str = Form('standard'),
    gas_correction: bool = Form(False),
    langley_cal: bool = Form(True),
    calc_mode: str = Form('fast')
):
    """提交AOD反演后台任务，立即返回task_id，前端轮询 /task-status/{task_id} 获取进度"""
    contents = await file.read()
    task_id = uuid.uuid4().hex[:12]
    task_store[task_id] = {'status': 'running', 'progress': 0.0, 'stage': 'Queued'}
    threading.Thread(
        target=_run_aod_task,
        args=(task_id, contents, file.filename, latitude, longitude, pressure,
              ozone_du, airmass_method, rayleigh_method, gas_correction,
              langley_cal, calc_mode),
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
                  langley_cal, calc_mode):
    """后台执行AOD反演，进度写入 task_store"""
    def set_prog(p, stage):
        task_store[task_id].update(progress=round(min(p, 99.0), 1), stage=stage)
    try:
        set_prog(2, 'Reading data file')
        dni_data = process_dni_data.read_dni_data(io.BytesIO(contents), original_filename=filename)
        # Langley定标（默认开启，失败波长自动回退Wehrli标准值）
        e0_langley = None
        e0_spectrum = None
        langley_plots = None
        if langley_cal:
            e0_langley, _, langley_plots = process_dni_data.langley_calibrate_day(
                dni_data, latitude=latitude, longitude=longitude, return_plot=True,
                progress_cb=lambda f: set_prog(5 + f * 10, 'Langley calibration (key wavelengths)'))
            # 全波段Langley定标（用于高光谱AOD曲线）
            e0_spectrum, _ = process_dni_data.langley_calibrate_spectrum(
                dni_data, latitude=latitude, longitude=longitude,
                airmass_method=airmass_method,
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

        result = {
            "time": [t.strftime('%H:%M') for t in aod_results['datetime']],
            "data_date": aod_results['datetime'].iloc[0].strftime('%Y-%m-%d'),
            "aod_440": aod_results['aod_440'].tolist(),
            "aod_500": aod_results['aod_500'].tolist(),
            "aod_675": aod_results['aod_675'].tolist(),
            "aod_870": aod_results['aod_870'].tolist(),
            "times": list(hyperspectral_data.keys()),
            "wavelengths": wavelengths,
            "wavelength_series": [data['aod'] for data in hyperspectral_data.values()],
            "wavelength_series_total": [data['optical_depths'] for data in hyperspectral_data.values()],
            "raw_view": _build_raw_view(dni_data, langley_plots)
        }
        # both 模式：附加向量化快速版结果供对比
        if calc_mode == 'both' and aod_fast is not None:
            result["aod_440_fast"] = aod_fast['aod_440'].tolist()
            result["aod_500_fast"] = aod_fast['aod_500'].tolist()
            result["aod_675_fast"] = aod_fast['aod_675'].tolist()
            result["aod_870_fast"] = aod_fast['aod_870'].tolist()

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
    pressure: float = Form(...),
    wv_method: str = Form('standard'),
    langley_cal: bool = Form(True),
    calc_mode: str = Form('fast'),
    pwv_a: float = Form(0.585),
    pwv_b: float = Form(0.569)
):
    """提交水汽反演后台任务，立即返回task_id，前端轮询 /task-status/{task_id} 获取进度"""
    contents = await file.read()
    task_id = uuid.uuid4().hex[:12]
    task_store[task_id] = {'status': 'running', 'progress': 0.0, 'stage': 'Queued'}
    threading.Thread(
        target=_run_wv_task,
        args=(task_id, contents, file.filename, latitude, longitude, pressure,
              wv_method, langley_cal, calc_mode, pwv_a, pwv_b),
        daemon=True
    ).start()
    return {"status": "started", "task_id": task_id}


def _run_wv_task(task_id, contents, filename, latitude, longitude, pressure,
                 wv_method, langley_cal, calc_mode, pwv_a=0.585, pwv_b=0.569):
    """后台执行水汽反演，进度写入 task_store"""
    def set_prog(p, stage):
        task_store[task_id].update(progress=round(min(p, 99.0), 1), stage=stage)
    try:
        set_prog(2, 'Reading data file')
        dni_data = process_dni_data.read_dni_data(io.BytesIO(contents), original_filename=filename)
        # Langley定标（默认开启，失败波长自动回退Wehrli标准值）
        e0_langley = None
        langley_plots = None
        if langley_cal:
            e0_langley, _, langley_plots = process_dni_data.langley_calibrate_day(
                dni_data, latitude=latitude, longitude=longitude, return_plot=True,
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
            "aod_870": wv_results['aod_870'].tolist(),
            "aod_1020": wv_results['aod_1020'].tolist(),
            "angstrom_alpha": wv_results['angstrom_alpha'].tolist(),
            "aod_baseline_936": wv_results['aod_baseline_936'].tolist(),
            "pwv_a": pwv_a,
            "pwv_b": pwv_b,
            "raw_view": _build_raw_view(dni_data, langley_plots)
        }
        # both 模式：附加向量化快速版结果供对比
        if calc_mode == 'both' and wv_fast is not None:
            result["wvod_936_fast"] = wv_fast['wvod_936'].tolist()
            result["pwv_mm_fast"] = wv_fast['pwv_mm'].tolist()
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
            # Save time series data
            if "time" in result and "aod_440" in result:
                time_series_data = {
                    "Time": result["time"],
                    "AOD 440nm": result["aod_440"],
                    "AOD 500nm": result["aod_500"],
                    "AOD 675nm": result["aod_675"],
                    "AOD 870nm": result["aod_870"]
                }
                time_series_df = pd.DataFrame(time_series_data)
                time_series_df.to_excel(writer, sheet_name="Time Series", index=False)

            # Save water vapor results
            if "time" in result and "wvod_936" in result:
                wv_data = {
                    "Time": result["time"],
                    "WVOD 936nm": result["wvod_936"],
                    "PWV (mm)": result.get("pwv_mm"),
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
    wavelength_chart: str = Form(None)
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