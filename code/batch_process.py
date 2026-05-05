import os
import sys
sys.path.insert(0, r'd:\卫星导航算法大作业\CPP-1')

from single_point_positioning_klobuchar_numpy import SPPSolver as SPPSolverKlobuchar
from single_point_positioning_no_ionosphere import SPPSolver as SPPSolverNoIono

def get_station_info(obs_file):
    """从观测文件名中提取测站名和GPS周信息"""
    basename = os.path.basename(obs_file)
    parts = basename.split('_')
    station_name = parts[0]

    # GPS周解析：从R_2024129格式中提取GPS周天
    gps_week_part = parts[2]  # 20241290000
    gps_week_day = int(gps_week_part[4:7])  # 129 or 132

    return station_name, gps_week_day

def get_approx_coords(station_name):
    """获取各测站的近似坐标"""
    coords_dict = {
        'DAV100ATA': [486854.546000, 2285099.292400, -5914955.713600],
        'HOB200AUS': [-3950072.249700, 2522415.361800, -4311637.402200],
        'KOUR00GUF': [3839591.433200, -5059567.551400, 579956.916400],
        'NTUS00SGP': [-1508022.572200, 6195577.395200, 148799.391200],
        'NYA200NOR': [1202379.310000, 252474.654300, 6237786.541700],
        'WTZR00DEU': [4075580.886300, 931853.578400, 4801567.970700],
    }
    return coords_dict.get(station_name, [0.0, 0.0, 0.0])

def process_single_file(obs_file, ephemeris_file, klobuchar_file, output_dir, station_name, gps_week_day, approx_coords, max_epochs=2880):
    """处理单个观测文件"""

    # Klobuchar模型版本
    solver_klobuchar = SPPSolverKlobuchar(obs_file, ephemeris_file, klobuchar_file, approx_coords)
    results_klobuchar = solver_klobuchar.process_file(max_epochs=max_epochs, debug=False)

    # 无电离层模型版本
    solver_no_iono = SPPSolverNoIono(obs_file, ephemeris_file, approx_coords)
    results_no_iono = solver_no_iono.process_file(max_epochs=max_epochs, debug=False)

    # 生成输出文件名，使用观测文件中的GPS周天
    output_file_klobuchar = os.path.join(output_dir, f"{station_name}_{gps_week_day}_spp_klobuchar.txt")
    output_file_no_iono = os.path.join(output_dir, f"{station_name}_{gps_week_day}_spp_no_ionosphere.txt")

    # 保存Klobuchar结果
    with open(output_file_klobuchar, 'w', encoding='utf-8') as f:
        f.write("GPS Single Point Positioning Results (with Klobuchar Ionosphere Model)\n")
        f.write("="*130 + "\n")
        f.write(f"Station: {station_name}\n")
        f.write(f"Approx Position: X={approx_coords[0]:.4f}, Y={approx_coords[1]:.4f}, Z={approx_coords[2]:.4f}\n")
        f.write(f"GPS Week Day: {gps_week_day}\n")
        f.write("\n")
        f.write(f"{'Epoch':<23} {'X (m)':>15} {'Y (m)':>15} {'Z (m)':>15} {'dX (m)':>10} {'dY (m)':>10} {'dZ (m)':>10} {'Clock (us)':>12} {'Satellites':>10} {'RMS (m)':>10} {'Max Res (m)':>12} {'Status':<30}\n")
        f.write("-"*130 + "\n")

        success_count = 0
        fail_count = 0

        for res in results_klobuchar:
            if res['failure_reason'] is not None:
                f.write(f"{res['epoch']:<23} {'-':>15} {'-':>15} {'-':>15} {'-':>10} {'-':>10} {'-':>10} {'-':>12} {res['num_satellites']:>10} {'-':>10} {'-':>12} {res['failure_reason']:<30}\n")
                fail_count += 1
            else:
                clock_us = (res['clock_offset'] / solver_klobuchar.c) * 1e6
                dx = res['x'] - approx_coords[0]
                dy = res['y'] - approx_coords[1]
                dz = res['z'] - approx_coords[2]
                rms = res['rms'] if res['rms'] is not None else 0.0
                max_res = res['max_residual'] if res['max_residual'] is not None else 0.0
                f.write(f"{res['epoch']:<23} {res['x']:>15.4f} {res['y']:>15.4f} {res['z']:>15.4f} {dx:>10.4f} {dy:>10.4f} {dz:>10.4f} {clock_us:>12.2f} {res['num_satellites']:>10} {rms:>10.4f} {max_res:>12.4f} {'Success':<30}\n")
                success_count += 1

        f.write("-"*130 + "\n")
        f.write(f"Summary: {success_count} successful epochs, {fail_count} failed epochs\n")

    # 保存无电离层模型结果
    with open(output_file_no_iono, 'w', encoding='utf-8') as f:
        f.write("GPS Single Point Positioning Results (without Ionosphere Correction)\n")
        f.write("="*130 + "\n")
        f.write(f"Station: {station_name}\n")
        f.write(f"Approx Position: X={approx_coords[0]:.4f}, Y={approx_coords[1]:.4f}, Z={approx_coords[2]:.4f}\n")
        f.write(f"GPS Week Day: {gps_week_day}\n")
        f.write("\n")
        f.write(f"{'Epoch':<23} {'X (m)':>15} {'Y (m)':>15} {'Z (m)':>15} {'dX (m)':>10} {'dY (m)':>10} {'dZ (m)':>10} {'Clock (us)':>12} {'Satellites':>10} {'RMS (m)':>10} {'Max Res (m)':>12} {'Status':<30}\n")
        f.write("-"*130 + "\n")

        success_count = 0
        fail_count = 0

        for res in results_no_iono:
            if res['failure_reason'] is not None:
                f.write(f"{res['epoch']:<23} {'-':>15} {'-':>15} {'-':>15} {'-':>10} {'-':>10} {'-':>10} {'-':>12} {res['num_satellites']:>10} {'-':>10} {'-':>12} {res['failure_reason']:<30}\n")
                fail_count += 1
            else:
                clock_us = (res['clock_offset'] / solver_no_iono.c) * 1e6
                dx = res['x'] - approx_coords[0]
                dy = res['y'] - approx_coords[1]
                dz = res['z'] - approx_coords[2]
                rms = res['rms'] if res['rms'] is not None else 0.0
                max_res = res['max_residual'] if res['max_residual'] is not None else 0.0
                f.write(f"{res['epoch']:<23} {res['x']:>15.4f} {res['y']:>15.4f} {res['z']:>15.4f} {dx:>10.4f} {dy:>10.4f} {dz:>10.4f} {clock_us:>12.2f} {res['num_satellites']:>10} {rms:>10.4f} {max_res:>12.4f} {'Success':<30}\n")
                success_count += 1

        f.write("-"*130 + "\n")
        f.write(f"Summary: {success_count} successful epochs, {fail_count} failed epochs\n")

    return output_file_klobuchar, output_file_no_iono, len(results_klobuchar), success_count, fail_count

def main():
    obs_data_dir = r'd:\卫星导航算法大作业\CPP-1\Data_01\obs_data'
    ephemeris_file = r'd:\卫星导航算法大作业\CPP-1\Data_01\ephemeris\brdc1290_ephemerides.csv'
    klobuchar_file = r'd:\卫星导航算法大作业\CPP-1\Data_01\ephemeris\brdc1290_klobuchar.csv'

    results_dir = r'd:\卫星导航算法大作业\CPP-1\Data_01\results'
    afternoon_dir = r'd:\卫星导航算法大作业\CPP-1\Data_01\afternoon'

    # 先删除输出目录中的所有spp结果文件
    print("清理旧文件...")
    for output_dir in [results_dir, afternoon_dir]:
        old_files = [f for f in os.listdir(output_dir) if f.endswith('.txt') and 'spp' in f]
        for f in old_files:
            try:
                os.remove(os.path.join(output_dir, f))
                print(f"  删除: {f}")
            except Exception as e:
                print(f"  删除失败 {f}: {e}")
    print("清理完成\n")

    obs_files = [f for f in os.listdir(obs_data_dir) if f.endswith('.csv')]

    print(f"找到 {len(obs_files)} 个观测文件")
    print("=" * 80)

    total_processed = 0
    total_success = 0
    total_failed = 0

    for obs_file in sorted(obs_files):
        obs_file_path = os.path.join(obs_data_dir, obs_file)
        station_name, gps_week_day = get_station_info(obs_file_path)
        approx_coords = get_approx_coords(station_name)

        print(f"\n处理文件: {obs_file}")
        print(f"测站: {station_name}, GPS周天: {gps_week_day}")
        print(f"近似坐标: X={approx_coords[0]:.4f}, Y={approx_coords[1]:.4f}, Z={approx_coords[2]:.4f}")

        # 处理并保存到 results 目录
        print(f"\n保存到 {results_dir}...")
        output_klobuchar, output_no_iono, total_epochs, success, failed = process_single_file(
            obs_file_path, ephemeris_file, klobuchar_file, results_dir,
            station_name, gps_week_day, approx_coords, max_epochs=2880
        )
        print(f"  Klobuchar模型: {output_klobuchar}")
        print(f"  无电离层模型: {output_no_iono}")
        print(f"  历元数: {total_epochs}, 成功: {success}, 失败: {failed}")

        total_processed += total_epochs
        total_success += success
        total_failed += failed

        # 处理并保存到 afternoon 目录
        print(f"\n保存到 {afternoon_dir}...")
        output_klobuchar, output_no_iono, total_epochs, success, failed = process_single_file(
            obs_file_path, ephemeris_file, klobuchar_file, afternoon_dir,
            station_name, gps_week_day, approx_coords, max_epochs=2880
        )
        print(f"  Klobuchar模型: {output_klobuchar}")
        print(f"  无电离层模型: {output_no_iono}")
        print(f"  历元数: {total_epochs}, 成功: {success}, 失败: {failed}")

        total_processed += total_epochs
        total_success += success
        total_failed += failed

    print("\n" + "=" * 80)
    print(f"批处理完成！")
    print(f"总计处理: {total_processed} 个历元")
    print(f"成功: {total_success}, 失败: {total_failed}")
    print(f"成功率: {total_success/total_processed*100:.2f}%")
    print("=" * 80)

if __name__ == '__main__':
    main()
