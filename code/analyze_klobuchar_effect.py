import os
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 脚本说明：Klobuchar 电离层模型改正效果分析脚本
# 功能：对比带/不带电离层改正的GPS单点定位结果，从多维度统计分析模型改善效果
# 输入：output/RESULT_morning 和 output/RESULT_afternoon 中的定位结果文件
# 输出：8个汇总CSV表 + 13幅分析图表
# =============================================================================

# 测站完整代码到简称的映射表
station_code_map = {
    "DAV100ATA": "DAV1",
    "HOB200AUS": "HOB2",
    "KOUR00GUF": "KOUR",
    "NTUS00SGP": "NTUS",
    "NYA200NOR": "NYA2",
    "WTZR00DEU": "WTZR"
}

# 测站元数据：包含每个测站的详细信息
stations = {
    "NYA2": {
        "full_code": "NYA200NOR",
        "name": "斯瓦尔巴",
        "hemisphere": "北半球",
        "latitude_band": "高纬",
        "lat": 78.93
    },
    "DAV1": {
        "full_code": "DAV100ATA",
        "name": "南极Davis站",
        "hemisphere": "南半球",
        "latitude_band": "高纬",
        "lat": -68.58
    },
    "WTZR": {
        "full_code": "WTZR00DEU",
        "name": "德国",
        "hemisphere": "北半球",
        "latitude_band": "中纬",
        "lat": 49.14
    },
    "HOB2": {
        "full_code": "HOB200AUS",
        "name": "霍巴特",
        "hemisphere": "南半球",
        "latitude_band": "中纬",
        "lat": -42.61
    },
    "NTUS": {
        "full_code": "NTUS00SGP",
        "name": "新加坡",
        "hemisphere": "北半球",
        "latitude_band": "低纬",
        "lat": 1.35
    },
    "KOUR": {
        "full_code": "KOUR00GUF",
        "name": "圭亚那",
        "hemisphere": "北半球",
        "latitude_band": "低纬",
        "lat": 5.25
    }
}

# 固定排序配置：用于保证统计汇总表和图表的展示顺序
LATITUDE_BAND_ORDER = ["高纬", "中纬", "低纬"]  # 纬度带顺序
HEMISPHERE_ORDER = ["北半球", "南半球"]          # 半球顺序
TIME_GROUP_ORDER = ["00", "12"]                # 时段顺序（00点、12点
STATION_ORDER = ["NYA2", "DAV1", "WTZR", "HOB2", "NTUS", "KOUR"]  # 测站顺序
DOY_ORDER = ["2024129", "2024132"]            # 年积日顺序

def setup_plot_style():
    """设置绘图风格为英文（用于生成国际通用的分析图表）"""
    sns.set_theme(style="whitegrid")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "DejaVu Sans"

# 中文到英文的映射表：用于生成英文图表
LATITUDE_BAND_MAP = {
    "高纬": "High Latitude",
    "中纬": "Mid Latitude",
    "低纬": "Low Latitude"
}

HEMISPHERE_MAP = {
    "北半球": "Northern Hemisphere",
    "南半球": "Southern Hemisphere"
}

TIME_GROUP_MAP = {
    "00": "00:00",
    "12": "12:00"
}

HEMI_LAT_MAP = {
    "北半球-高纬": "Northern Hemisphere - High Latitude",
    "南半球-高纬": "Southern Hemisphere - High Latitude",
    "北半球-中纬": "Northern Hemisphere - Mid Latitude",
    "南半球-中纬": "Southern Hemisphere - Mid Latitude",
    "北半球-低纬": "Northern Hemisphere - Low Latitude",
    "南半球-低纬": "Southern Hemisphere - Low Latitude"
}

LATITUDE_ORDER_EN = ["High Latitude", "Mid Latitude", "Low Latitude"]
HEMISPHERE_ORDER_EN = ["Northern Hemisphere", "Southern Hemisphere"]
TIME_ORDER_EN = ["00:00", "12:00"]
HEMI_LAT_ORDER_EN = [
    "Northern Hemisphere - High Latitude",
    "Southern Hemisphere - High Latitude",
    "Northern Hemisphere - Mid Latitude",
    "Southern Hemisphere - Mid Latitude",
    "Northern Hemisphere - Low Latitude",
    "Southern Hemisphere - Low Latitude"
]

def parse_filename(file_name, time_group):
    """
    从文件名解析测站信息、年积日、模型类型和时段
    
    参数：
        file_name (str): 定位结果文件名
        time_group (str): 时段标识 ("00" 或 "12")
    
    返回：
        dict: 包含解析后信息的字典
    """
    try:
        # 移除文件扩展名
        base_name = os.path.splitext(file_name)[0]
        parts = base_name.split('_')
        
        if len(parts) >= 4:
            station_full_code = parts[0]  # 测站完整代码
            doy = parts[1]                 # 年积日
            model_part = '_'.join(parts[2:])  # 模型部分
            
            # 获取测站简称
            if station_full_code in station_code_map:
                station = station_code_map[station_full_code]
            else:
                warnings.warn(f"无法识别的测站代码: {station_full_code}")
                station = station_full_code
            
            # 解析模型类型（Klobuchar或无电离层改正
            if 'spp_klobuchar' in model_part:
                model = 'klobuchar'
            elif 'spp_no_ionosphere' in model_part:
                model = 'no_ionosphere'
            else:
                warnings.warn(f"无法识别的模型类型: {model_part}")
                model = 'unknown'
            
            return {
                'station_full_code': station_full_code,
                'station': station,
                'doy': doy,
                'model': model,
                'time_group': time_group
            }
        else:
            warnings.warn(f"无法解析文件名: {file_name}")
            return None
    except Exception as e:
        warnings.warn(f"解析文件名 {file_name} 时出错: {str(e)}")
        return None

def parse_spp_result_file(file_path):
    """
    解析单个 SPP 结果 txt 文件，提取每个历元的定位数据
    
    参数：
        file_path (str): SPP 结果文件路径
    
    返回：
        tuple: (DataFrame, int, int) - 历元记录、成功历元数、失败历元数
    """
    records = []
    success_count = 0
    fail_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # 跳过空行
                if not line:
                    continue
                
                # 跳过表头和分隔线
                if line.startswith('GPS Single Point') or \
                   line.startswith('=') or \
                   line.startswith('Station:') or \
                   line.startswith('Approx Position:') or \
                   line.startswith('GPS Week:') or \
                   line.startswith('Epoch') or \
                   line.startswith('-'):
                    continue
                
                # 尝试解析 Summary 行获取成功/失败历元数
                if line.startswith('Summary:'):
                    match = re.search(r'(\d+) successful epochs?, (\d+) failed epochs?', line)
                    if match:
                        success_count = int(match.group(1))
                        fail_count = int(match.group(2))
                    continue
                
                # 检查是否为失败行
                if any(keyword in line.lower() for keyword in ['fail', 'error', 'insufficient']):
                    fail_count += 1
                    continue
                
                # 检查是否以日期格式开头（这是有效历元行）
                if re.match(r'^\d{4}-\d{2}-\d{2}', line):
                    parts = line.split()
                    
                    if len(parts) >= 11:
                        try:
                            # 解析各字段
                            epoch = parts[0] + " " + parts[1]
                            x = float(parts[2])
                            y = float(parts[3])
                            z = float(parts[4])
                            dx = float(parts[5])
                            dy = float(parts[6])
                            dz = float(parts[7])
                            clock = float(parts[8])
                            satellites = int(parts[9])
                            rms = float(parts[10])
                            
                            max_res = float(parts[11]) if len(parts) > 11 else np.nan
                            status = ' '.join(parts[12:]) if len(parts) > 12 else ''
                            
                            # 保存记录
                            records.append({
                                'epoch': epoch,
                                'x': x,
                                'y': y,
                                'z': z,
                                'dx': dx,
                                'dy': dy,
                                'dz': dz,
                                'clock': clock,
                                'satellites': satellites,
                                'rms': rms,
                                'max_res': max_res,
                                'status': status,
                                'success': True
                            })
                            success_count += 1
                        except ValueError as e:
                            fail_count += 1
                            warnings.warn(f"解析历元行失败: {line}, 错误: {str(e)}")
                    else:
                        fail_count += 1
                        warnings.warn(f"历元行字段不足: {line}")
    
    except Exception as e:
        warnings.warn(f"读取文件 {file_path} 时出错: {str(e)}")
    
    return pd.DataFrame(records), success_count, fail_count

def compute_file_stats(records, station_info):
    """
    对单个文件的成功历元计算统计量
    
    参数：
        records (DataFrame): 历元数据
        station_info (dict): 测站元数据
    
    返回：
        dict: 包含统计量的字典
    """
    stats = {
        'success_epochs': len(records),      # 成功历元数
        'failed_epochs': 0,                  # 失败历元数
        'mean_rms': np.nan,                  # 平均残差RMS
        'std_rms': np.nan,                   # RMS标准差
        'mean_max_res': np.nan,              # 平均最大残差
        'mean_satellites': np.nan,           # 平均卫星数
        'mean_dx': np.nan,                   # 平均dx偏差
        'mean_dy': np.nan,                   # 平均dy偏差
        'mean_dz': np.nan,                   # 平均dz偏差
        'mean_abs_dx': np.nan,               # 平均|dx|绝对值
        'mean_abs_dy': np.nan,               # 平均|dy|绝对值
        'mean_abs_dz': np.nan,               # 平均|dz|绝对值
        'mean_d3d': np.nan                   # 平均3D位置误差
    }
    
    if len(records) > 0:
        stats['mean_rms'] = records['rms'].mean()
        stats['std_rms'] = records['rms'].std()
        stats['mean_max_res'] = records['max_res'].mean()
        stats['mean_satellites'] = records['satellites'].mean()
        stats['mean_dx'] = records['dx'].mean()
        stats['mean_dy'] = records['dy'].mean()
        stats['mean_dz'] = records['dz'].mean()
        stats['mean_abs_dx'] = records['dx'].abs().mean()
        stats['mean_abs_dy'] = records['dy'].abs().mean()
        stats['mean_abs_dz'] = records['dz'].abs().mean()
        stats['mean_d3d'] = np.sqrt(records['dx']**2 + records['dy']**2 + records['dz']**2).mean()
    
    # 添加测站信息
    stats.update({
        'station_name': station_info.get('name', ''),
        'hemisphere': station_info.get('hemisphere', ''),
        'latitude_band': station_info.get('latitude_band', ''),
        'lat': station_info.get('lat', np.nan)
    })
    
    return stats

def build_file_level_dataframe(morning_dir, afternoon_dir):
    """
    批量读取所有定位结果 txt 文件，生成文件级统计 DataFrame
    
    参数：
        morning_dir (str): 00点时段结果目录
        afternoon_dir (str): 12点时段结果目录
    
    返回：
        tuple: (DataFrame, int, int, int) - 文件统计、总文件数、成功文件数、失败文件数
    """
    all_stats = []
    total_files = 0
    success_files = 0
    fail_files = 0
    
    def process_directory(directory, time_group_label):
        """内部函数：处理单个目录中的所有文件"""
        nonlocal total_files, success_files, fail_files
        
        if not os.path.exists(directory):
            warnings.warn(f"目录不存在: {directory}")
            return
        
        print(f"正在读取 {directory}...")
        
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                total_files += 1
                file_path = os.path.join(directory, filename)
                
                # 解析文件名获取基本信息
                parsed = parse_filename(filename, time_group_label)
                if parsed is None:
                    fail_files += 1
                    continue
                
                station = parsed['station']
                if station not in stations:
                    warnings.warn(f"未知测站: {station}")
                    fail_files += 1
                    continue
                
                station_info = stations[station]
                
                # 解析文件内容获取历元数据
                df_records, success_epochs, failed_epochs = parse_spp_result_file(file_path)
                
                # 计算该文件的统计量
                stats = compute_file_stats(df_records, station_info)
                
                # 添加文件信息
                stats.update({
                    'file_name': filename,
                    'file_path': file_path,
                    'station_full_code': parsed['station_full_code'],
                    'station': station,
                    'doy': parsed['doy'],
                    'time_group': parsed['time_group'],
                    'model': parsed['model'],
                    'failed_epochs': failed_epochs
                })
                
                all_stats.append(stats)
                success_files += 1
                
                # 打印解析进度
                print(f"  文件: {filename}")
                print(f"    测站: {station} ({station_info['name']}), 年积日: {parsed['doy']}")
                print(f"    模型: {parsed['model']}, 时段: {parsed['time_group']}点")
                print(f"    成功历元: {success_epochs}, 失败历元: {failed_epochs}")
                print()
    
    # 处理两个时段目录
    process_directory(morning_dir, '00')
    process_directory(afternoon_dir, '12')
    
    # 打印汇总信息
    print(f"\n读取完成:")
    print(f"  总文件数: {total_files}")
    print(f"  成功解析文件数: {success_files}")
    print(f"  未成功解析文件数: {fail_files}")
    
    return pd.DataFrame(all_stats), total_files, success_files, fail_files

def build_paired_comparison(df_file):
    """
    将同站、同年积日、同时段的 klobuchar 和 no_ionosphere 文件配对，计算改善效果
    
    参数：
        df_file (DataFrame): 文件级统计数据
    
    返回：
        tuple: (DataFrame, int, int) - 配对比较数据、配对数、缺失配对数
    """
    paired_data = []
    missing_pairs = 0
    
    # 按测站 + 年积日 + 时段分组，确保是同一组对比
    groups = df_file.groupby(['station', 'doy', 'time_group'])
    
    for (station, doy, time_group), group in groups:
        # 获取两个模型的数据
        klobuchar_data = group[group['model'] == 'klobuchar']
        no_iono_data = group[group['model'] == 'no_ionosphere']
        
        # 检查是否都有数据
        if len(klobuchar_data) == 0:
            warnings.warn(f"缺少 klobuchar 数据: {station}_{doy}_{time_group}")
            missing_pairs += 1
            continue
        
        if len(no_iono_data) == 0:
            warnings.warn(f"缺少 no_ionosphere 数据: {station}_{doy}_{time_group}")
            missing_pairs += 1
            continue
        
        klobuchar = klobuchar_data.iloc[0]
        no_iono = no_iono_data.iloc[0]
        
        # 计算改善指标
        rms_no_iono = no_iono['mean_rms']
        rms_klobuchar = klobuchar['mean_rms']
        d3d_no_iono = no_iono['mean_d3d']
        d3d_klobuchar = klobuchar['mean_d3d']
        
        delta_rms = rms_no_iono - rms_klobuchar  # RMS改善量（正数表示变好）
        delta_d3d = d3d_no_iono - d3d_klobuchar  # 3D位置改善量
        
        # 计算改善率（百分比
        rms_improvement_rate = (delta_rms / rms_no_iono) * 100 if rms_no_iono not in [0, np.nan] else np.nan
        d3d_improvement_rate = (delta_d3d / d3d_no_iono) * 100 if d3d_no_iono not in [0, np.nan] else np.nan
        
        paired_data.append({
            'station': station,
            'station_full_code': klobuchar['station_full_code'],
            'station_name': klobuchar['station_name'],
            'hemisphere': klobuchar['hemisphere'],
            'latitude_band': klobuchar['latitude_band'],
            'lat': klobuchar['lat'],
            'doy': doy,
            'time_group': time_group,
            'rms_no_iono': rms_no_iono,
            'rms_klobuchar': rms_klobuchar,
            'delta_rms': delta_rms,
            'rms_improvement_rate': rms_improvement_rate,
            'd3d_no_iono': d3d_no_iono,
            'd3d_klobuchar': d3d_klobuchar,
            'delta_d3d': delta_d3d,
            'd3d_improvement_rate': d3d_improvement_rate,
            'satellites_no_iono': no_iono['mean_satellites'],
            'satellites_klobuchar': klobuchar['mean_satellites'],
            'success_epochs_no_iono': no_iono['success_epochs'],
            'success_epochs_klobuchar': klobuchar['success_epochs']
        })
    
    # 打印配对结果
    print(f"\n成对比较完成:")
    print(f"  成对比较数量: {len(paired_data)}")
    print(f"  缺失配对数量: {missing_pairs}")
    
    return pd.DataFrame(paired_data), len(paired_data), missing_pairs

def save_summary_tables(df_file, df_paired, output_dir):
    """
    保存所有 8 个 CSV 汇总表
    
    参数：
        df_file (DataFrame): 文件级统计数据
        df_paired (DataFrame): 配对比较数据
        output_dir (str): 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. file_level_stats.csv - 每个文件的详细统计
    file_level_path = os.path.join(output_dir, 'file_level_stats.csv')
    df_file.to_csv(file_level_path, encoding='utf-8-sig', index=False)
    print(f"已保存: {file_level_path}")
    
    # 2. paired_comparison.csv - 同站同年积日同时段的配对比较
    paired_path = os.path.join(output_dir, 'paired_comparison.csv')
    df_paired.to_csv(paired_path, encoding='utf-8-sig', index=False)
    print(f"已保存: {paired_path}")
    
    # 3. latitude_band_summary.csv - 按纬度带汇总统计
    lat_band_summary = df_paired.groupby('latitude_band').agg(
        sample_count=('delta_rms', 'count'),
        mean_delta_rms=('delta_rms', 'mean'),
        std_delta_rms=('delta_rms', 'std'),
        mean_rms_improvement_rate=('rms_improvement_rate', 'mean'),
        mean_delta_d3d=('delta_d3d', 'mean'),
        std_delta_d3d=('delta_d3d', 'std'),
        mean_d3d_improvement_rate=('d3d_improvement_rate', 'mean')
    ).reindex(LATITUDE_BAND_ORDER).reset_index()
    lat_band_summary_path = os.path.join(output_dir, 'latitude_band_summary.csv')
    lat_band_summary.to_csv(lat_band_summary_path, encoding='utf-8-sig', index=False)
    print(f"已保存: {lat_band_summary_path}")
    
    # 4. hemisphere_summary.csv - 按南北半球汇总统计
    hemi_summary = df_paired.groupby('hemisphere').agg(
        sample_count=('delta_rms', 'count'),
        mean_delta_rms=('delta_rms', 'mean'),
        std_delta_rms=('delta_rms', 'std'),
        mean_rms_improvement_rate=('rms_improvement_rate', 'mean'),
        mean_delta_d3d=('delta_d3d', 'mean'),
        std_delta_d3d=('delta_d3d', 'std'),
        mean_d3d_improvement_rate=('d3d_improvement_rate', 'mean')
    ).reindex(HEMISPHERE_ORDER).reset_index()
    hemi_summary_path = os.path.join(output_dir, 'hemisphere_summary.csv')
    hemi_summary.to_csv(hemi_summary_path, encoding='utf-8-sig', index=False)
    print(f"已保存: {hemi_summary_path}")
    
    # 5. hemi_lat_summary.csv - 按半球和纬度带组合汇总统计
    hemi_lat_summary = df_paired.groupby(['hemisphere', 'latitude_band']).agg(
        sample_count=('delta_rms', 'count'),
        mean_delta_rms=('delta_rms', 'mean'),
        std_delta_rms=('delta_rms', 'std'),
        mean_rms_improvement_rate=('rms_improvement_rate', 'mean'),
        mean_delta_d3d=('delta_d3d', 'mean'),
        std_delta_d3d=('delta_d3d', 'std'),
        mean_d3d_improvement_rate=('d3d_improvement_rate', 'mean')
    ).reset_index()
    hemi_lat_summary_path = os.path.join(output_dir, 'hemi_lat_summary.csv')
    hemi_lat_summary.to_csv(hemi_lat_summary_path, encoding='utf-8-sig', index=False)
    print(f"已保存: {hemi_lat_summary_path}")
    
    # 6. time_group_summary.csv - 按时段（00/12点）汇总统计
    time_summary = df_paired.groupby('time_group').agg(
        sample_count=('delta_rms', 'count'),
        mean_delta_rms=('delta_rms', 'mean'),
        std_delta_rms=('delta_rms', 'std'),
        mean_rms_improvement_rate=('rms_improvement_rate', 'mean'),
        mean_delta_d3d=('delta_d3d', 'mean'),
        std_delta_d3d=('delta_d3d', 'std'),
        mean_d3d_improvement_rate=('d3d_improvement_rate', 'mean')
    ).reindex(TIME_GROUP_ORDER).reset_index()
    time_summary_path = os.path.join(output_dir, 'time_group_summary.csv')
    time_summary.to_csv(time_summary_path, encoding='utf-8-sig', index=False)
    print(f"已保存: {time_summary_path}")
    
    # 7. station_summary.csv - 按测站汇总统计
    station_summary = df_paired.groupby('station').agg(
        station_name=('station_name', 'first'),
        hemisphere=('hemisphere', 'first'),
        latitude_band=('latitude_band', 'first'),
        sample_count=('delta_rms', 'count'),
        mean_delta_rms=('delta_rms', 'mean'),
        std_delta_rms=('delta_rms', 'std'),
        mean_rms_improvement_rate=('rms_improvement_rate', 'mean'),
        mean_delta_d3d=('delta_d3d', 'mean'),
        std_delta_d3d=('delta_d3d', 'std'),
        mean_d3d_improvement_rate=('d3d_improvement_rate', 'mean')
    ).reindex(STATION_ORDER).reset_index()
    station_summary_path = os.path.join(output_dir, 'station_summary.csv')
    station_summary.to_csv(station_summary_path, encoding='utf-8-sig', index=False)
    print(f"已保存: {station_summary_path}")
    
    # 8. doy_time_summary.csv - 按年积日和时段汇总统计
    doy_time_summary = df_paired.groupby(['doy', 'time_group']).agg(
        sample_count=('delta_rms', 'count'),
        mean_delta_rms=('delta_rms', 'mean'),
        std_delta_rms=('delta_rms', 'std'),
        mean_rms_improvement_rate=('rms_improvement_rate', 'mean'),
        mean_delta_d3d=('delta_d3d', 'mean'),
        std_delta_d3d=('delta_d3d', 'std'),
        mean_d3d_improvement_rate=('d3d_improvement_rate', 'mean')
    ).reset_index()
    doy_time_summary_path = os.path.join(output_dir, 'doy_time_summary.csv')
    doy_time_summary.to_csv(doy_time_summary_path, encoding='utf-8-sig', index=False)
    print(f"已保存: {doy_time_summary_path}")
    
    print(f"\n所有 CSV 文件已保存到 {output_dir}")

def plot_latitude_hemisphere_effect(df_paired, output_dir):
    """
    绘制纬度和半球相关的 6 幅分析图像（全部使用英文）
    
    参数：
        df_paired (DataFrame): 配对比较数据
        output_dir (str): 输出目录
    """
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    setup_plot_style()
    
    # 准备绘图数据，添加英文映射
    df_plot = df_paired.copy()
    df_plot["latitude_band_en"] = df_plot["latitude_band"].map(LATITUDE_BAND_MAP)
    df_plot["hemisphere_en"] = df_plot["hemisphere"].map(HEMISPHERE_MAP)
    df_plot["hemi_lat"] = df_plot["hemisphere"] + "-" + df_plot["latitude_band"]
    df_plot["hemi_lat_en"] = df_plot["hemi_lat"].map(HEMI_LAT_MAP)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 图1：按纬度带的RMS改善
    plt.figure(figsize=(10, 6))
    lat_band_data = df_plot.groupby('latitude_band_en')['delta_rms'].agg(['mean', 'std'])
    lat_band_data = lat_band_data.reindex([x for x in LATITUDE_ORDER_EN if x in lat_band_data.index])
    if lat_band_data.empty:
        warnings.warn("No valid data for fig_01. Skipped.")
    else:
        plt.bar(lat_band_data.index, lat_band_data['mean'], yerr=lat_band_data['std'], capsize=5, color=colors[0])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('RMS Improvement by Latitude Band Using Klobuchar Model')
        plt.xlabel('Latitude Band')
        plt.ylabel('RMS Improvement, $\\Delta$RMS / m')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig_01_latitude_band_delta_rms.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 图2：按半球的RMS改善
    plt.figure(figsize=(10, 6))
    hemi_data = df_plot.groupby('hemisphere_en')['delta_rms'].agg(['mean', 'std'])
    hemi_data = hemi_data.reindex([x for x in HEMISPHERE_ORDER_EN if x in hemi_data.index])
    if hemi_data.empty:
        warnings.warn("No valid data for fig_02. Skipped.")
    else:
        plt.bar(hemi_data.index, hemi_data['mean'], yerr=hemi_data['std'], capsize=5, color=colors[1])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('RMS Improvement by Hemisphere Using Klobuchar Model')
        plt.xlabel('Hemisphere')
        plt.ylabel('RMS Improvement, $\\Delta$RMS / m')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig_02_hemisphere_delta_rms.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. fig_03_hemi_lat_delta_rms.png
    plt.figure(figsize=(12, 6))
    hemi_lat_data = df_plot.groupby('hemi_lat_en')['delta_rms'].agg(['mean', 'std'])
    valid_order = [x for x in HEMI_LAT_ORDER_EN if x in hemi_lat_data.index]
    hemi_lat_data = hemi_lat_data.reindex(valid_order)
    if hemi_lat_data.empty:
        warnings.warn("No valid data for fig_03. Skipped.")
    else:
        plt.bar(hemi_lat_data.index, hemi_lat_data['mean'], yerr=hemi_lat_data['std'], capsize=5, color=colors[2])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('RMS Improvement by Hemisphere and Latitude Band Using Klobuchar Model')
        plt.xlabel('Hemisphere-Latitude Band')
        plt.ylabel('RMS Improvement, $\\Delta$RMS / m')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig_03_hemi_lat_delta_rms.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. fig_04_station_rms_improvement_rate.png
    plt.figure(figsize=(10, 6))
    station_data = df_plot.groupby('station')['rms_improvement_rate'].mean().reindex(STATION_ORDER)
    plt.bar(station_data.index, station_data.values, color=colors[3])
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Station-wise RMS Improvement Rate Using Klobuchar Model')
    plt.xlabel('Station')
    plt.ylabel('RMS Improvement Rate / %')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_04_station_rms_improvement_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. fig_05_latitude_band_delta_d3d.png
    plt.figure(figsize=(10, 6))
    lat_band_d3d = df_plot.groupby('latitude_band_en')['delta_d3d'].agg(['mean', 'std'])
    lat_band_d3d = lat_band_d3d.reindex([x for x in LATITUDE_ORDER_EN if x in lat_band_d3d.index])
    if lat_band_d3d.empty:
        warnings.warn("No valid data for fig_05. Skipped.")
    else:
        plt.bar(lat_band_d3d.index, lat_band_d3d['mean'], yerr=lat_band_d3d['std'], capsize=5, color=colors[4])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('3D Position Improvement by Latitude Band Using Klobuchar Model')
        plt.xlabel('Latitude Band')
        plt.ylabel('3D Position Improvement, $\\Delta$d3D / m')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig_05_latitude_band_delta_d3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. fig_06_station_delta_d3d.png
    plt.figure(figsize=(10, 6))
    station_d3d = df_plot.groupby('station')['delta_d3d'].mean().reindex(STATION_ORDER)
    plt.bar(station_d3d.index, station_d3d.values, color=colors[5])
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Station-wise 3D Position Improvement Using Klobuchar Model')
    plt.xlabel('Station')
    plt.ylabel('3D Position Improvement, $\\Delta$d3D / m')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_06_station_delta_d3d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Latitude and hemisphere figures saved to {fig_dir}")

def plot_time_effect(df_paired, output_dir):
    """
    绘制时段相关的 7 幅分析图像（全部使用英文）
    
    参数：
        df_paired (DataFrame): 配对比较数据
        output_dir (str): 输出目录
    """
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    setup_plot_style()
    
    # 准备绘图数据，添加英文映射
    df_plot = df_paired.copy()
    df_plot["latitude_band_en"] = df_plot["latitude_band"].map(LATITUDE_BAND_MAP)
    df_plot["time_group_en"] = df_plot["time_group"].map(TIME_GROUP_MAP)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 图7：按时段的RMS改善
    plt.figure(figsize=(10, 6))
    time_data = df_plot.groupby('time_group_en')['delta_rms'].agg(['mean', 'std'])
    time_data = time_data.reindex([x for x in TIME_ORDER_EN if x in time_data.index])
    if time_data.empty:
        warnings.warn("No valid data for fig_07. Skipped.")
    else:
        plt.bar(time_data.index, time_data['mean'], yerr=time_data['std'], capsize=5, color=colors[0])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('RMS Improvement at 00:00 and 12:00 Using Klobuchar Model')
        plt.xlabel('Time Group')
        plt.ylabel('RMS Improvement, $\\Delta$RMS / m')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig_07_time_delta_rms.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 8. fig_08_time_delta_d3d.png
    plt.figure(figsize=(10, 6))
    time_d3d = df_plot.groupby('time_group_en')['delta_d3d'].agg(['mean', 'std'])
    time_d3d = time_d3d.reindex([x for x in TIME_ORDER_EN if x in time_d3d.index])
    if time_d3d.empty:
        warnings.warn("No valid data for fig_08. Skipped.")
    else:
        plt.bar(time_d3d.index, time_d3d['mean'], yerr=time_d3d['std'], capsize=5, color=colors[1])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('3D Position Improvement at 00:00 and 12:00 Using Klobuchar Model')
        plt.xlabel('Time Group')
        plt.ylabel('3D Position Improvement, $\\Delta$d3D / m')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig_08_time_delta_d3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 9. fig_09_station_time_rms_improvement_rate.png
    plt.figure(figsize=(12, 6))
    station_time_data = df_plot.groupby(['station', 'time_group_en'])['rms_improvement_rate'].mean().unstack().reindex(STATION_ORDER)
    valid_cols = [x for x in TIME_ORDER_EN if x in station_time_data.columns]
    station_time_data = station_time_data[valid_cols]
    if station_time_data.empty or station_time_data.dropna().empty:
        warnings.warn("No valid data for fig_09. Skipped.")
    else:
        station_time_data.plot(kind='bar', ax=plt.gca(), width=0.8, color=colors[:len(valid_cols)])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Station-wise RMS Improvement Rate at 00:00 and 12:00')
        plt.xlabel('Station')
        plt.ylabel('RMS Improvement Rate / %')
        plt.legend(title='Time')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig_09_station_time_rms_improvement_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 10. fig_10_latitude_time_delta_rms.png
    plt.figure(figsize=(10, 6))
    lat_time_data = df_plot.groupby(['latitude_band_en', 'time_group_en'])['delta_rms'].mean().unstack()
    valid_lat = [x for x in LATITUDE_ORDER_EN if x in lat_time_data.index]
    lat_time_data = lat_time_data.reindex(valid_lat)
    valid_cols = [x for x in TIME_ORDER_EN if x in lat_time_data.columns]
    lat_time_data = lat_time_data[valid_cols]
    if lat_time_data.empty or lat_time_data.dropna().empty:
        warnings.warn("No valid data for fig_10. Skipped.")
    else:
        lat_time_data.plot(kind='bar', ax=plt.gca(), width=0.8, color=colors[:len(valid_cols)])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('RMS Improvement by Latitude Band at 00:00 and 12:00')
        plt.xlabel('Latitude Band')
        plt.ylabel('RMS Improvement, $\\Delta$RMS / m')
        plt.legend(title='Time')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig_10_latitude_time_delta_rms.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 11. fig_11_doy_time_delta_rms.png
    plt.figure(figsize=(10, 6))
    doy_time_data = df_plot.groupby(['doy', 'time_group_en'])['delta_rms'].mean().unstack().reindex(DOY_ORDER)
    valid_cols = [x for x in TIME_ORDER_EN if x in doy_time_data.columns]
    doy_time_data = doy_time_data[valid_cols]
    if doy_time_data.empty or doy_time_data.dropna().empty:
        warnings.warn("No valid data for fig_11. Skipped.")
    else:
        doy_time_data.plot(kind='bar', ax=plt.gca(), width=0.8, color=colors[:len(valid_cols)])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('RMS Improvement by Day of Year at 00:00 and 12:00')
        plt.xlabel('Day of Year')
        plt.ylabel('RMS Improvement, $\\Delta$RMS / m')
        plt.legend(title='Time')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig_11_doy_time_delta_rms.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 12. fig_12_station_time_delta_d3d.png
    plt.figure(figsize=(12, 6))
    station_time_d3d = df_plot.groupby(['station', 'time_group_en'])['delta_d3d'].mean().unstack().reindex(STATION_ORDER)
    valid_cols = [x for x in TIME_ORDER_EN if x in station_time_d3d.columns]
    station_time_d3d = station_time_d3d[valid_cols]
    if station_time_d3d.empty or station_time_d3d.dropna().empty:
        warnings.warn("No valid data for fig_12. Skipped.")
    else:
        station_time_d3d.plot(kind='bar', ax=plt.gca(), width=0.8, color=colors[:len(valid_cols)])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Station-wise 3D Position Improvement at 00:00 and 12:00')
        plt.xlabel('Station')
        plt.ylabel('3D Position Improvement, $\\Delta$d3D / m')
        plt.legend(title='Time')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig_12_station_time_delta_d3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 13. fig_13_latitude_time_delta_d3d.png
    plt.figure(figsize=(10, 6))
    lat_time_d3d = df_plot.groupby(['latitude_band_en', 'time_group_en'])['delta_d3d'].mean().unstack()
    valid_lat = [x for x in LATITUDE_ORDER_EN if x in lat_time_d3d.index]
    lat_time_d3d = lat_time_d3d.reindex(valid_lat)
    valid_cols = [x for x in TIME_ORDER_EN if x in lat_time_d3d.columns]
    lat_time_d3d = lat_time_d3d[valid_cols]
    if lat_time_d3d.empty or lat_time_d3d.dropna().empty:
        warnings.warn("No valid data for fig_13. Skipped.")
    else:
        lat_time_d3d.plot(kind='bar', ax=plt.gca(), width=0.8, color=colors[:len(valid_cols)])
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('3D Position Improvement by Latitude Band at 00:00 and 12:00')
        plt.xlabel('Latitude Band')
        plt.ylabel('3D Position Improvement, $\\Delta$d3D / m')
        plt.legend(title='Time')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'fig_13_latitude_time_delta_d3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Time effect figures saved to {fig_dir}")

def main():
    """
    主函数：执行完整的 Klobuchar 电离层模型改正效果分析流程
    
    工作流程：
    1. 读取所有定位结果文件
    2. 构建同站同年积日同时段的配对比较
    3. 生成并保存 8 个汇总统计表
    4. 绘制 13 幅分析图像
    """
    # 设置路径
    morning_dir = r"../output/RESULT_morning"
    afternoon_dir = r"../output/RESULT_afternoon"
    output_dir = r"../output"
    
    # 检查输入目录是否存在
    if not os.path.exists(morning_dir):
        print(f"错误：输入目录不存在: {morning_dir}")
        return
    if not os.path.exists(afternoon_dir):
        print(f"错误：输入目录不存在: {afternoon_dir}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Klobuchar 电离层模型改正效果分析程序")
    print("=" * 60)
    
    # 1. 读取所有文件并生成文件级统计
    df_file, total_files, success_files, fail_files = build_file_level_dataframe(morning_dir, afternoon_dir)
    
    # 2. 构建成对比较（同站同年积日同时段）
    df_paired, paired_count, missing_count = build_paired_comparison(df_file)
    
    # 3. 保存汇总表格
    save_summary_tables(df_file, df_paired, output_dir)
    
    # 4. 绘制纬度和半球影响图（6幅）
    plot_latitude_hemisphere_effect(df_paired, output_dir)
    
    # 5. 绘制时段影响图（7幅）
    plot_time_effect(df_paired, output_dir)
    
    print("\n" + "=" * 60)
    print("分析完成：已输出统计表和图像到 output 文件夹")
    print("=" * 60)

if __name__ == '__main__':
    main()