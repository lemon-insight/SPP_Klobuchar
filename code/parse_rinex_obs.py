# -*- coding: utf-8 -*-
"""
RINEX观测文件解析器

本模块实现了RINEX 3.x格式GNSS观测文件的解析功能，主要用于提取GPS卫星的C1C伪距观测值。
解析结果包括：
1. 文件头信息（版本、测站名、近似坐标、观测类型等）
2. 历元数据（时间、可见卫星列表、伪距观测值）

输出格式：
- 可读报告文件（_report.txt）：用于人工查看
- 元数据文件（_meta.txt）：用于单点定位初始化
- 观测数据文件（_obs.csv）：CSV格式，便于单点定位计算

技术特点：
- 支持RINEX 3.x版本
- 自动识别GPS卫星（PRN 1-32）
- 提取C1C伪距和S1C信号强度
- 自动计算GPS周和周内秒

作者：卫星导航算法课程作业组
日期：2024年5月
"""

import os
from datetime import datetime

class RinexObsParser:
    """
    RINEX观测文件解析器
    
    解析RINEX 3.x格式的GNSS观测文件，提取GPS卫星的伪距观测值。
    主要功能：
    - 解析文件头（版本、测站名、近似坐标、观测类型）
    - 解析观测历元（时间、卫星列表、伪距值）
    - 计算GPS周和周内秒
    - 输出多种格式的解析结果
    """
    
    # GPS时间起始点（1980年1月6日0时0分0秒）
    # GPS时间系统与UTC不同，从此时刻开始计数，没有闰秒
    GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
    
    def __init__(self, file_path):
        """
        初始化RINEX观测文件解析器
        
        参数：
            file_path: str - RINEX观测文件路径
        """
        self.file_path = file_path      # RINEX文件路径
        self.lines = []                 # 文件内容行列表
        self.header = {                 # 头文件信息
            'version': '',              # RINEX版本
            'marker_name': '',          # 测站名
            'approx_position_xyz': [],  # 近似坐标（ITRF XYZ）
            'obs_types': {},            # 按系统分类的观测类型（如{'G': ['C1C', 'L1C', ...]}）
            'sys_obs_count': {}         # 各系统观测类型数量
        }
        self.epochs = []                # 观测历元数据列表

    def parse(self):
        """
        解析整个RINEX观测文件
        
        流程：
        1. 读取文件内容
        2. 解析文件头信息
        3. 解析观测数据
        4. 返回结构化结果
        
        返回：
            dict - 包含header和epochs的解析结果
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        
        self._parse_header()       # 解析文件头
        self._parse_observations() # 解析观测数据
        return self._get_results() # 返回结果

    def _parse_header(self):
        """
        解析RINEX文件头信息
        
        文件头包含以下关键信息：
        - RINEX版本和类型
        - 测站名称（MARKER NAME）
        - 测站近似坐标（APPROX POSITION XYZ）
        - 各系统的观测类型（SYS / # / OBS TYPES）
        
        解析直到遇到'END OF HEADER'标志
        """
        in_header = True
        
        for line in self.lines:
            if not in_header:
                break
                
            # 检测文件头结束
            if 'END OF HEADER' in line:
                in_header = False
                continue
            
            # 解析版本信息（前20个字符）
            if 'RINEX VERSION / TYPE' in line:
                self.header['version'] = line[:20].strip()
            
            # 解析测站名称（前60个字符）
            elif 'MARKER NAME' in line:
                self.header['marker_name'] = line[:60].strip()
            
            # 解析近似坐标（前60个字符）
            elif 'APPROX POSITION XYZ' in line:
                parts = line[:60].split()
                if len(parts) >= 3:
                    self.header['approx_position_xyz'] = [
                        float(parts[0]),  # X坐标
                        float(parts[1]),  # Y坐标
                        float(parts[2])   # Z坐标
                    ]
            
            # 解析观测类型
            elif 'SYS / # / OBS TYPES' in line:
                sys_code = line[0].strip()  # 系统代码（G=GPS, C=BDS, E=Galileo等）
                if sys_code:
                    content = line[3:60].strip()
                    parts = content.split()
                    if parts:
                        count = int(parts[0])      # 观测类型数量
                        obs_types = parts[1:]      # 观测类型列表
                        
                        if sys_code not in self.header['obs_types']:
                            self.header['obs_types'][sys_code] = []
                        self.header['obs_types'][sys_code].extend(obs_types)
                        self.header['sys_obs_count'][sys_code] = count

    def _parse_observations(self):
        """
        解析观测数据部分
        
        观测数据由一系列历元组成，每个历元以'>'开头，包含：
        1. 历元时间（年、月、日、时、分、秒）
        2. 可见卫星数
        3. 每颗卫星的观测值
        
        遍历文件，遇到'>'开头的行时调用_parse_epoch解析单个历元
        """
        i = 0
        while i < len(self.lines):
            line = self.lines[i]
            
            # 历元数据以'>'开头
            if line.startswith('>'):
                epoch_data = self._parse_epoch(line, i)
                if epoch_data:
                    self.epochs.append(epoch_data)
                    # 跳过已解析的卫星行
                    i += len(epoch_data['satellites']) + 1
                else:
                    i += 1
            else:
                i += 1

    def _compute_gps_week(self, year, month, day, hour, minute, second):
        """
        从日历时间计算GPS周和周内秒
        
        GPS时间系统：
        - GPS周：从GPS纪元（1980年1月6日）开始的完整周数
        - 周内秒：当前周内的秒数（0 ~ 604800秒）
        
        参数：
            year, month, day, hour, minute, second: int/float - 日历时间
        
        返回：
            tuple - (gps_week, gps_week_seconds)
        """
        current = datetime(year, month, day, hour, minute, int(second), int((second - int(second)) * 1e6))
        delta = current - self.GPS_EPOCH
        total_seconds = delta.total_seconds()
        
        gps_week = int(total_seconds / (7 * 86400))       # GPS周（一周=7*24*3600=604800秒）
        gps_week_seconds = total_seconds % (7 * 86400)    # 周内秒（0~604800）
        
        return gps_week, gps_week_seconds

    def _parse_epoch(self, epoch_line, start_idx):
        """
        解析单个历元数据
        
        参数：
            epoch_line: str - 历元头行（以'>'开头）
            start_idx: int - 当前行索引
        
        返回：
            dict or None - 包含时间信息和卫星数据的历元字典，解析失败返回None
        """
        try:
            parts = epoch_line.split()
            if len(parts) < 8:
                return None

            # 提取时间信息
            year = int(parts[1])
            month = int(parts[2])
            day = int(parts[3])
            hour = int(parts[4])
            minute = int(parts[5])
            second = float(parts[6])

            # 计算GPS周和周内秒
            gps_week, gps_week_seconds = self._compute_gps_week(year, month, day, hour, minute, second)

            # 组织时间信息
            time_info = {
                'year': year,
                'month': month,
                'day': day,
                'hour': hour,
                'minute': minute,
                'second': second,
                'gps_week': gps_week,
                'gps_week_seconds': gps_week_seconds,
                'num_satellites': int(parts[8])  # 可见卫星数
            }
            
            satellites = []
            i = start_idx + 1  # 从下一行开始读取卫星数据
            
            # 获取GPS系统的观测类型列表，找到C1C和S1C的索引
            gps_obs_types = self.header['obs_types'].get('G', [])
            c1_index = None   # C1C伪距索引（用于单点定位）
            s1_index = None   # S1C信号强度索引（用于质量控制）
            
            # 遍历观测类型列表，找到所需观测值的位置
            for idx, obs_type in enumerate(gps_obs_types):
                if obs_type in ['C1C', 'C1']:      # C1C或C1伪距（C/A码伪距）
                    c1_index = idx
                elif obs_type in ['S1C', 'S1']:    # S1C或S1信号强度（载噪比）
                    s1_index = idx
            
            # 读取卫星数据行
            while i < len(self.lines) and len(satellites) < time_info['num_satellites']:
                line = self.lines[i].strip()
                if not line or line.startswith('>'):
                    break  # 到达下一个历元或文件结束
                
                line_parts = line.split()
                if not line_parts:
                    i += 1
                    continue
                
                sat_id = line_parts[0]  # 卫星标识（如G01, G12等）
                if not sat_id:
                    i += 1
                    continue
                
                # 只处理GPS卫星（以G开头）
                if sat_id.startswith('G'):
                    prn = int(sat_id[1:])  # 提取PRN号（如G01 -> 1）
                    
                    # 解析观测值
                    values = []
                    for val_str in line_parts[1:]:
                        try:
                            values.append(float(val_str))
                        except ValueError:
                            values.append(None)  # 无效值用None表示
                    
                    # 提取C1C伪距
                    c1_value = None
                    if c1_index is not None and c1_index < len(values):
                        c1_value = values[c1_index]
                    
                    # 提取S1C信号强度
                    s1_value = None
                    if s1_index is not None and s1_index < len(values):
                        s1_value = values[s1_index]
                    
                    # 添加卫星数据
                    satellites.append({
                        'sat_id': sat_id,       # 卫星标识（如G01）
                        'prn': prn,             # PRN号
                        'c1_value': c1_value,   # C1C伪距（米）
                        's1_value': s1_value,   # S1C信号强度（dBHz）
                        'raw_values': values    # 原始观测值列表
                    })
                
                i += 1
            
            return {
                'time': time_info,            # 时间信息
                'satellites': satellites,     # 卫星数据列表
                'total_satellites': time_info['num_satellites']  # 总卫星数
            }
        
        except Exception as e:
            print(f"解析历元时出错: {e}")
            return None

    def _get_results(self):
        """
        返回结构化的解析结果
        
        返回：
            dict - 包含header和epochs的字典
        """
        return {
            'header': {
                'version': self.header['version'],
                'marker_name': self.header['marker_name'],
                'approx_position_xyz': self.header['approx_position_xyz'],
                'observation_types': self.header['obs_types'],
                'system_observation_counts': self.header['sys_obs_count']
            },
            'epochs': self.epochs
        }

def format_metadata(results, input_file):
    """
    生成元数据文件内容 - 适合单点定位初始化
    
    参数：
        results: dict - 解析结果
        input_file: str - 输入文件名
    
    返回：
        str - 格式化的元数据内容
    """
    output = []
    h = results['header']
    
    # 文件标识
    output.append('GNSS_OBS_METADATA')
    output.append('=' * 60)
    
    # 基本信息
    output.append(f'FILE_NAME: {os.path.basename(input_file)}')
    output.append(f'RINEX_VERSION: {h["version"]}')
    output.append(f'MARKER_NAME: {h["marker_name"]}')
    output.append('')
    
    # 测站近似坐标（用于单点定位初始值）
    output.append('STATION_COORDINATES_XYZ')
    output.append('------------------------')
    xyz = h['approx_position_xyz']
    output.append(f'X: {xyz[0]:.6f}')
    output.append(f'Y: {xyz[1]:.6f}')
    output.append(f'Z: {xyz[2]:.6f}')
    output.append('')
    
    # GPS观测类型
    output.append('GPS_OBSERVATION_TYPES')
    output.append('---------------------')
    gps_types = h['observation_types'].get('G', [])
    for obs_type in gps_types:
        output.append(f'{obs_type}')
    output.append('')
    
    # 统计信息
    output.append('STATISTICS')
    output.append('----------')
    output.append(f'TOTAL_EPOCHS: {len(results["epochs"])}')
    
    # 第一个历元信息
    if results['epochs']:
        first_epoch = results['epochs'][0]['time']
        output.append(f'FIRST_EPOCH: {first_epoch["year"]:04d}-{first_epoch["month"]:02d}-{first_epoch["day"]:02d} {first_epoch["hour"]:02d}:{first_epoch["minute"]:02d}:{first_epoch["second"]:.6f}')
        output.append(f'FIRST_EPOCH_GPS_WEEK: {first_epoch["gps_week"]}')
        output.append(f'FIRST_EPOCH_VISIBLE_SATELLITES: {first_epoch["num_satellites"]}')
    
    return '\n'.join(output)

def format_observation_data(results):
    """
    生成观测数据文件内容 - 适合单点定位计算（CSV格式）
    
    参数：
        results: dict - 解析结果
    
    返回：
        str - CSV格式的观测数据
    """
    output = []
    
    # CSV表头
    output.append('epoch_time,year,month,day,hour,minute,second,gps_week,gps_week_seconds,prn,c1_meters,s1_dBHz')
    
    # 遍历每个历元
    for epoch in results['epochs']:
        t = epoch['time']
        
        # 格式化时间字符串
        time_str = f"{t['year']:04d}-{t['month']:02d}-{t['day']:02d} {t['hour']:02d}:{t['minute']:02d}:{t['second']:.6f}"
        
        # 遍历每颗卫星
        for sat in epoch['satellites']:
            c1_val = sat['c1_value'] if sat['c1_value'] is not None else 'NaN'
            s1_val = sat['s1_value'] if sat['s1_value'] is not None else 'NaN'
            
            # 输出一行CSV数据
            output.append(
                f'{time_str},{t["year"]},{t["month"]},{t["day"]},{t["hour"]},{t["minute"]},{t["second"]:.6f},{t["gps_week"]},{t["gps_week_seconds"]},{sat["prn"]},{c1_val},{s1_val}'
            )
    
    return '\n'.join(output)

def format_readable_output(results):
    """
    生成人类可读的输出格式（报告形式）
    
    参数：
        results: dict - 解析结果
    
    返回：
        str - 格式化的可读报告
    """
    output = []
    h = results['header']
    
    output.append('=' * 80)
    output.append('RINEX 观测文件解析结果 (GNSS单频伪距单点定位专用)')
    output.append('=' * 80)
    output.append('')
    
    # 1. 测站信息
    output.append('【1】测站信息')
    output.append('-' * 60)
    output.append(f'  测站名: {h["marker_name"]}')
    output.append(f'  RINEX版本: {h["version"]}')
    output.append('')
    
    # 2. 测站近似坐标
    output.append('【2】测站近似坐标 (ITRF XYZ)')
    output.append('-' * 60)
    xyz = h['approx_position_xyz']
    output.append(f'  X = {xyz[0]:.4f} m')
    output.append(f'  Y = {xyz[1]:.4f} m')
    output.append(f'  Z = {xyz[2]:.4f} m')
    output.append('')
    
    # 3. GPS观测类型
    output.append('【3】GPS 观测类型')
    output.append('-' * 60)
    gps_types = h['observation_types'].get('G', [])
    output.append(f'  GPS系统观测类型 ({len(gps_types)}个):')
    output.append(f'    {", ".join(gps_types)}')
    output.append('')
    
    # 4. 数据概览
    output.append('【4】数据概览')
    output.append('-' * 60)
    output.append(f'  历元总数: {len(results["epochs"])}')
    if results['epochs']:
        first_epoch = results['epochs'][0]['time']
        last_epoch = results['epochs'][-1]['time']
        first_time = f"{first_epoch['year']:04d}-{first_epoch['month']:02d}-{first_epoch['day']:02d} {first_epoch['hour']:02d}:{first_epoch['minute']:02d}:{first_epoch['second']:.3f}"
        last_time = f"{last_epoch['year']:04d}-{last_epoch['month']:02d}-{last_epoch['day']:02d} {last_epoch['hour']:02d}:{last_epoch['minute']:02d}:{last_epoch['second']:.3f}"
        output.append(f'  时间范围: {first_time} 至 {last_time}')
        
        # 统计可见卫星数
        total_sat_count = sum(epoch['total_satellites'] for epoch in results['epochs'])
        avg_sat = total_sat_count / len(results['epochs']) if results['epochs'] else 0
        output.append(f'  平均可见卫星数: {avg_sat:.1f}')
    output.append('')
    
    # 5. 示例数据（前2个历元）
    output.append('【5】示例数据 (前2个历元)')
    output.append('-' * 60)
    
    if not results['epochs']:
        output.append('  未找到观测历元数据')
    else:
        for idx, epoch in enumerate(results['epochs'][:2]):
            t = epoch['time']
            time_str = f"{t['year']:04d}-{t['month']:02d}-{t['day']:02d} {t['hour']:02d}:{t['minute']:02d}:{t['second']:.3f}"
            output.append(f'\n  历元 {idx+1}: {time_str}')
            output.append(f'  GPS周: {t["gps_week"]}, GPS卫星数: {len(epoch["satellites"])}')
            output.append('')
            output.append('    PRN   C1伪距(m)    S1强度(dBHz)')
            output.append('    -------------------------------')
            
            for sat in epoch['satellites']:
                c1_str = f'{sat["c1_value"]:.3f}' if sat['c1_value'] is not None else '-----'
                s1_str = f'{sat["s1_value"]:.1f}' if sat['s1_value'] is not None else '---'
                output.append(f'    G{sat["prn"]:02d}   {c1_str:>12}    {s1_str:>11}')
    
    output.append('')
    output.append('=' * 80)
    output.append('提示: 详细数据请查看 .obs 和 .meta 文件')
    
    return '\n'.join(output)

def main():
    """
    主函数：批量处理RINEX观测文件
    
    流程：
    1. 扫描指定目录下的所有.rnx文件
    2. 对每个文件进行解析
    3. 生成三种输出文件：
       - _report.txt: 人类可读报告
       - _meta.txt: 元数据（含测站坐标）
       - _obs.csv: 观测数据CSV
    """
    obs_dir = r'd:\卫星导航算法大作业\obs'        # RINEX文件目录
    output_dir = r'd:\卫星导航算法大作业\CPP-1\Data_01'  # 输出目录
    
    # 检查输入目录是否存在
    if not os.path.exists(obs_dir):
        print(f"错误：目录不存在: {obs_dir}")
        return
    
    # 查找所有.rnx文件
    rnx_files = []
    for root, dirs, files in os.walk(obs_dir):
        for file in files:
            if file.endswith('.rnx'):
                rnx_files.append(os.path.join(root, file))
    
    if not rnx_files:
        print("未找到任何 .rnx 文件")
        return
    
    print(f"找到 {len(rnx_files)} 个 RINEX 观测文件")
    print('=' * 80)
    
    # 批量处理每个文件
    for input_file in rnx_files:
        try:
            print(f"\n正在处理: {os.path.basename(input_file)}")
            
            # 创建解析器并解析
            parser = RinexObsParser(input_file)
            results = parser.parse()
            
            # 生成基础文件名（去掉.rnx扩展名）
            base_name = os.path.basename(input_file).replace('.rnx', '')
            
            # 1. 生成可读报告文件
            readable_output = format_readable_output(results)
            readable_file = os.path.join(output_dir, f'{base_name}_report.txt')
            with open(readable_file, 'w', encoding='utf-8') as f:
                f.write(readable_output)
            print(f'  报告文件: {readable_file}')
            
            # 2. 生成元数据文件（用于定位初始化）
            metadata_output = format_metadata(results, input_file)
            metadata_file = os.path.join(output_dir, f'{base_name}_meta.txt')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                f.write(metadata_output)
            print(f'  元数据文件: {metadata_file}')
            
            # 3. 生成观测数据文件（CSV格式，便于单点定位计算）
            obs_output = format_observation_data(results)
            obs_file = os.path.join(output_dir, f'{base_name}_obs.csv')
            with open(obs_file, 'w', encoding='utf-8') as f:
                f.write(obs_output)
            print(f'  观测数据文件: {obs_file}')
            
        except Exception as e:
            print(f"处理文件 {input_file} 时出错: {e}")
    
    print('\n' + '=' * 80)
    print(f"批量处理完成！共处理 {len(rnx_files)} 个文件")
    print("每个文件生成了3个输出文件:")
    print("  - _report.txt: 人类可读的报告")
    print("  - _meta.txt:   元数据（含测站坐标）")
    print("  - _obs.csv:    观测数据CSV（便于单点定位计算）")

if __name__ == '__main__':
    main()
