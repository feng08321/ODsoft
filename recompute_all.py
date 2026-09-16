# -*- coding: utf-8 -*-
"""
用修正后的算法（赤纬BUG已修复，QX/T 69-2024标准公式）重算全部svd数据
产品：7波长AOD + Ångström指数 + 936nm水汽光学厚度（WVOD）
"""
import os
import glob
import time
import traceback
import pandas as pd
import process_dni_data as p

DATA_DIR = r'20211011svd'
OUT_DIR = r'20260916重算结果_Langley'
LAT, LON, PRES = 49.0, 119.4, 840.0

os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(DATA_DIR, '*svd*.xlsx')))
print(f"共 {len(files)} 个数据文件")

summary = []
for fi, fpath in enumerate(files):
    name = os.path.basename(fpath)
    t0 = time.time()
    print(f"\n[{fi+1}/{len(files)}] {name}")
    try:
        d = p.read_dni_data(fpath, original_filename=name)

        # Langley 定标：按天回归各波长大气顶层信号 E0
        e0_langley, langley_rep = p.langley_calibrate_day(d, latitude=LAT, longitude=LON)
        langley_rep.to_excel(os.path.join(OUT_DIR, name.replace('.xlsx', '_langley.xlsx')), index=False)
        print(f"  Langley定标成功波长: {list(e0_langley.keys())}")

        # AOD 反演（7个特征波长，气体扣除默认屏蔽，使用Langley定标E0）
        aod = p.process_dni_data(d, latitude=LAT, longitude=LON, pressure=PRES,
                                 e0_langley=e0_langley)

        # 936nm 水汽反演（QX/T 69-2024 标准基线法，使用Langley定标E0）
        wv = p.process_water_vapor(d, latitude=LAT, longitude=LON, pressure=PRES,
                                   wv_method='standard', e0_langley=e0_langley)

        # 合并（水汽表只取水汽相关列，避免 aod_870 等列重复）
        wv_cols = wv[['datetime', 'aod_1020', 'aod_baseline_936', 'wvod_936']]
        merged = pd.merge(aod, wv_cols, on='datetime', how='left')

        out_path = os.path.join(OUT_DIR, name.replace('.xlsx', '_results.xlsx'))
        merged.to_excel(out_path, index=False)

        n_aod500 = int(aod['aod_500'].gt(0).sum())
        n_wv = int(wv['wvod_936'].notna().sum())
        elapsed = time.time() - t0
        summary.append({
            'file': name,
            'status': 'ok',
            'n_total': len(merged),
            'n_langley_wl': len(e0_langley),
            'n_aod500_valid': n_aod500,
            'n_wvod_valid': n_wv,
            'aod500_mean': aod['aod_500'].mean(),
            'wvod_mean': wv['wvod_936'].mean(),
            'seconds': round(elapsed, 1)
        })
        print(f"  完成: AOD500有效{n_aod500}, WVOD有效{n_wv}, 用时{elapsed:.0f}s -> {out_path}")
    except Exception as e:
        summary.append({'file': name, 'status': f'failed: {e}'})
        print(f"  失败: {e}")
        traceback.print_exc()

summary_df = pd.DataFrame(summary)
summary_path = os.path.join(OUT_DIR, 'summary.xlsx')
summary_df.to_excel(summary_path, index=False)
print(f"\n全部完成，汇总: {summary_path}")
print(summary_df.to_string(index=False))
