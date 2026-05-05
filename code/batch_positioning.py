import os
import sys
sys.path.insert(0, r'd:\卫星导航算法大作业\CPP-1')

from single_point_positioning_no_ionosphere import SPPSolver as SPPNoIono
from single_point_positioning_klobuchar_numpy import SPPSolver as SPPKlobuchar

DATA_DIR = r'd:\卫星导航算法大作业\CPP-1\Data_01'
OBS_DIR = os.path.join(DATA_DIR, 'obs_data')
EPH_DIR = os.path.join(DATA_DIR, 'ephemeris')
RES_DIR = os.path.join(DATA_DIR, 'results')

OBS_FILES = [
    'DAV100ATA_R_20241290000_01D_30S_MO_obs.csv',
    'HOB200AUS_R_20241290000_01D_30S_MO_obs.csv',
    'KOUR00GUF_R_20241290000_01D_30S_MO_obs.csv',
    'NTUS00SGP_R_20241290000_01D_30S_MO_obs.csv',
    'NYA200NOR_R_20241290000_01D_30S_MO_obs.csv',
    'WTZR00DEU_R_20241290000_01D_30S_MO_obs.csv',
    'DAV100ATA_R_20241320000_01D_30S_MO_obs.csv',
    'HOB200AUS_R_20241320000_01D_30S_MO_obs.csv',
    'KOUR00GUF_R_20241320000_01D_30S_MO_obs.csv',
    'NTUS00SGP_R_20241320000_01D_30S_MO_obs.csv',
    'NYA200NOR_R_20241320000_01D_30S_MO_obs.csv',
    'WTZR00DEU_R_20241320000_01D_30S_MO_obs.csv',
]

NAV_FILES = {
    '2024129': ('brdc1290_ephemerides.csv', 'brdc1290_klobuchar.csv'),
    '2024132': ('brdc1320_ephemerides.csv', 'brdc1320_klobuchar.csv'),
}

STATION_COORDS = {
    'DAV100ATA': [486854.546000, 2285099.292400, -5914955.713600],
    'HOB200AUS': [-3950072.249700, 2522415.361800, -4311637.402200],
    'KOUR00GUF': [3839591.433200, -5059567.551400, 579956.916400],
    'NTUS00SGP': [-1508022.572200, 6195577.395200, 148799.391200],
    'NYA200NOR': [1202379.310000, 252474.654300, 6237786.541700],
    'WTZR00DEU': [4075580.886300, 931853.578400, 4801567.970700],
}

def get_station_name(filename):
    return filename.split('_')[0]

def get_obs_day(filename):
    if '2024129' in filename:
        return '2024129'
    elif '2024132' in filename:
        return '2024132'
    return None

def process_no_ionosphere(obs_file, ephemeris_file, station_name):
    print(f"  [无电离层模型] 处理中...")
    obs_path = os.path.join(OBS_DIR, obs_file)
    eph_path = os.path.join(EPH_DIR, ephemeris_file)

    if station_name in STATION_COORDS:
        station_coords = STATION_COORDS[station_name]
    else:
        station_coords = [0.0, 0.0, 0.0]
        print(f"    警告: 未找到测站 {station_name} 的近似坐标")

    solver = SPPNoIono(obs_path, eph_path, station_coords)
    results = solver.process_file(max_epochs=100)

    return solver, results

def process_klobuchar(obs_file, ephemeris_file, klobuchar_file, station_name):
    print(f"  [Klobuchar模型] 处理中...")
    obs_path = os.path.join(OBS_DIR, obs_file)
    eph_path = os.path.join(EPH_DIR, ephemeris_file)
    klb_path = os.path.join(EPH_DIR, klobuchar_file)

    if station_name in STATION_COORDS:
        station_coords = STATION_COORDS[station_name]
    else:
        station_coords = [0.0, 0.0, 0.0]
        print(f"    警告: 未找到测站 {station_name} 的近似坐标")

    solver = SPPKlobuchar(obs_path, eph_path, klb_path, station_coords)
    results = solver.process_file(max_epochs=100)

    return solver, results

def save_results(results, solver, output_file, model_name):
    approx_coords = solver.station_coords
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"GPS Single Point Positioning Results ({model_name})\n")
        f.write("=" * 100 + "\n")
        f.write(f"Station: {os.path.basename(output_file).split('_')[0]}\n")
        f.write(f"Approx Position: X={approx_coords[0]:.4f}, Y={approx_coords[1]:.4f}, Z={approx_coords[2]:.4f}\n")
        if hasattr(solver, 'gps_week'):
            f.write(f"GPS Week: {solver.gps_week}\n")
        f.write("\n")
        f.write(f"{'Epoch':<23} {'X (m)':>15} {'Y (m)':>15} {'Z (m)':>15} {'dX (m)':>10} {'dY (m)':>10} {'dZ (m)':>10} {'Clock (us)':>12} {'Satellites':>10} {'RMS (m)':>10}\n")
        f.write("-" * 100 + "\n")

        for res in results:
            clock_us = (res['clock_offset'] / solver.c) * 1e6
            dx = res['x'] - approx_coords[0]
            dy = res['y'] - approx_coords[1]
            dz = res['z'] - approx_coords[2]
            rms = res.get('rms', 0.0)
            f.write(f"{res['epoch']:<23} {res['x']:>15.4f} {res['y']:>15.4f} {res['z']:>15.4f} {dx:>10.4f} {dy:>10.4f} {dz:>10.4f} {clock_us:>12.2f} {res['num_satellites']:>10} {rms:>10.4f}\n")

def main():
    print("=" * 80)
    print("GPS单频伪距单点定位 - 批量处理")
    print("=" * 80)

    total_files = len(OBS_FILES)
    for idx, obs_file in enumerate(OBS_FILES, 1):
        station_name = get_station_name(obs_file)
        obs_day = get_obs_day(obs_file)

        if obs_day is None:
            print(f"\n[{idx}/{total_files}] 无法确定观测日期: {obs_file}")
            continue

        ephemeris_file, klobuchar_file = NAV_FILES[obs_day]

        print(f"\n[{idx}/{total_files}] 处理: {obs_file}")
        print(f"  测站: {station_name}, 观测日: {obs_day}")

        try:
            solver_no_iono, results_no_iono = process_no_ionosphere(
                obs_file, ephemeris_file, station_name)

            result_file_no_iono = os.path.join(
                RES_DIR,
                f"{station_name}_{obs_day}_spp_no_ionosphere.txt"
            )
            save_results(results_no_iono, solver_no_iono, result_file_no_iono,
                        "without Ionosphere Correction")
            print(f"    已保存: {os.path.basename(result_file_no_iono)} ({len(results_no_iono)} 历元)")

        except Exception as e:
            print(f"    错误 [无电离层模型]: {e}")

        try:
            solver_klobuchar, results_klobuchar = process_klobuchar(
                obs_file, ephemeris_file, klobuchar_file, station_name)

            result_file_klobuchar = os.path.join(
                RES_DIR,
                f"{station_name}_{obs_day}_spp_klobuchar.txt"
            )
            save_results(results_klobuchar, solver_klobuchar, result_file_klobuchar,
                        "with Klobuchar Ionosphere Model")
            print(f"    已保存: {os.path.basename(result_file_klobuchar)} ({len(results_klobuchar)} 历元)")

        except Exception as e:
            print(f"    错误 [Klobuchar模型]: {e}")

    print("\n" + "=" * 80)
    print("批量处理完成!")
    print("=" * 80)

if __name__ == '__main__':
    main()
