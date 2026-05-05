# -*- coding: utf-8 -*-
"""
GPS单点定位算法实现（带Klobuchar电离层改正）

本模块实现了基于伪距观测值的GPS单点定位算法，使用Klobuchar模型进行电离层延迟改正。
算法流程：
1. 读取观测文件和星历文件
2. 对每个历元进行以下处理：
   - 筛选GPS卫星（PRN 1-32）
   - 检查伪距质量（范围：1e7 ~ 5e7米）
   - 计算卫星位置和钟差
   - 计算电离层延迟（Klobuchar模型）
   - 计算对流层延迟（Saastamoinen模型简化版）
   - 建立观测方程并求解
   - 粗差剔除（迭代剔除残差最大的卫星）
3. 输出定位结果

主要技术参数：
- GM = 3.986005e14 m^3/s^2 (地球引力常数)
- omega_e = 7.2921151467e-5 rad/s (地球自转角速度)
- c = 299792458.0 m/s (光速)
- GPS-UTC = 18秒（2024年闰秒）

作者：卫星导航算法课程作业组
日期：2024年5月
"""

import os
import math
from datetime import datetime
import numpy as np

class SPPSolver:
    """
    GPS单点定位求解器（带Klobuchar电离层改正）
    
    使用伪距观测值进行单点定位，包含以下主要功能：
    - 卫星星历加载和管理
    - Klobuchar电离层参数加载
    - GPS时间系统处理（支持GPS/UTC时间转换）
    - 卫星位置计算（考虑地球自转改正）
    - 卫星钟差计算（含相对论改正）
    - 电离层延迟计算（Klobuchar模型）
    - 对流层延迟计算（简化模型）
    - 加权最小二乘求解
    - 粗差剔除（迭代法）
    """
    
    def __init__(self, obs_file, ephemeris_file, klobuchar_file, station_coords=None, debug=False, time_system='GPS'):
        """
        初始化单点定位求解器
        
        参数：
            obs_file: str - 观测数据CSV文件路径
            ephemeris_file: str - 卫星星历CSV文件路径
            klobuchar_file: str - Klobuchar电离层参数CSV文件路径
            station_coords: list - 接收机近似坐标 [X, Y, Z]（ITRF，单位：米）
            debug: bool - 是否启用调试模式（默认False）
            time_system: str - 时间系统（'GPS' 或 'UTC'，默认'GPS'）
        """
        self.obs_file = obs_file
        self.ephemeris_file = ephemeris_file
        self.klobuchar_file = klobuchar_file
        self.station_coords = station_coords
        self.debug = debug
        self.time_system = time_system.upper()  # 时间系统: 'GPS' 或 'UTC'

        # 物理常数
        self.GM = 3.986005e14          # 地球引力常数 (m^3/s^2)
        self.omega_e = 7.2921151467e-5 # 地球自转角速度 (rad/s)
        self.c = 299792458.0           # 光速 (m/s)
        self.week_sec = 604800.0       # GPS周秒数 (7*24*3600)
        self.F = -4.442807633e-10      # 相对论改正系数 (-2*sqrt(GM)/c^3)
        
        # GPS-UTC 闰秒（2024年为18秒，根据需要可更新）
        self.gps_utc_offset = 18.0

        # Klobuchar电离层参数
        self.alpha = [0.0, 0.0, 0.0, 0.0]  # 振幅系数
        self.beta = [0.0, 0.0, 0.0, 0.0]   # 周期系数
        self.ephemerides = {}               # 卫星星历字典 {prn: [eph1, eph2, ...]}
        self.gps_week = 2313               # GPS周

        # 加载数据文件
        self._load_klobuchar()
        self._load_ephemerides()

    def _week_wrap(self, tk):
        """
        周归化函数
        
        将时间差限制在[-302400, 302400]秒范围内（即±GPS周的一半）
        用于处理GPS周边界问题
        
        参数：
            tk: float - 时间差（秒）
        
        返回：
            float - 周归化后的时间差
        """
        while tk > self.week_sec / 2:
            tk -= self.week_sec
        while tk < -self.week_sec / 2:
            tk += self.week_sec
        return tk

    def _load_klobuchar(self):
        """
        加载Klobuchar电离层参数
        
        从CSV文件中读取alpha和beta系数，用于电离层延迟计算
        文件格式：param_name, value
        例如：alpha0, 0.123456e-08
        """
        with open(self.klobuchar_file, 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:  # 跳过表头
            parts = line.strip().split(',')
            param = parts[0]
            value = float(parts[1])
            if param.startswith('alpha'):
                idx = int(param[-1])
                self.alpha[idx] = value
            elif param.startswith('beta'):
                idx = int(param[-1])
                self.beta[idx] = value

    def _load_ephemerides(self):
        """
        加载卫星广播星历
        
        从CSV文件中读取卫星星历参数，存储为字典结构
        支持多颗卫星的多个星历片段
        """
        with open(self.ephemeris_file, 'r') as f:
            lines = f.readlines()

        header = lines[0].strip().split(',')
        for line in lines[1:]:  # 跳过表头
            parts = line.strip().split(',')
            prn = int(parts[0])
            eph = {}
            for i, key in enumerate(header):
                try:
                    eph[key] = float(parts[i])
                except:
                    eph[key] = 0.0
            eph['prn'] = int(eph['prn'])
            eph['week'] = int(eph['week'])

            if prn not in self.ephemerides:
                self.ephemerides[prn] = []
            self.ephemerides[prn].append(eph)

            # 更新最大GPS周数
            if eph['week'] > self.gps_week:
                self.gps_week = eph['week']

    def _compute_gpst(self, year, month, day, hour, minute, second):
        """
        将日历时间转换为GPS周内秒
        
        根据时间系统标识决定是否需要加闰秒：
        - GPS时间：直接计算，不需要加闰秒
        - UTC时间：需要加上GPS-UTC闰秒
        
        参数：
            year, month, day, hour, minute, second: int/float - 日历时间
        
        返回：
            float - GPS周内秒（0 ~ 604800）
        """
        # GPS时间起始点：1980年1月6日0时0分0秒
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
        current = datetime(year, month, day, hour, minute, int(second), int((second - int(second)) * 1e6))
        delta = current - gps_epoch
        total_seconds = delta.total_seconds()
        
        # 根据时间系统决定是否需要加闰秒
        # GPS时间 = UTC时间 + GPS-UTC闰秒
        if self.time_system == 'UTC':
            total_seconds += self.gps_utc_offset
            if self.debug:
                print(f"时间系统: UTC, 添加闰秒 {self.gps_utc_offset}s")
        elif self.time_system == 'GPS':
            # GPS时间不需要加闰秒
            if self.debug:
                print(f"时间系统: GPS, 无需加闰秒")
        else:
            # 默认按GPS处理
            if self.debug:
                print(f"时间系统: {self.time_system}, 默认按GPS处理")
        
        gpst = total_seconds % self.week_sec
        
        if self.debug:
            print(f"观测时间: {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:.3f}")
            print(f"GPS周内秒 tr: {gpst:.3f} s")
        
        return gpst

    def _find_ephemeris(self, prn, time):
        """
        查找最优卫星星历
        
        根据卫星PRN和观测时间，查找最合适的星历片段
        使用周归化时间差进行匹配，选择时间差最小且小于7200秒的星历
        
        参数：
            prn: int - 卫星PRN号
            time: float - GPS周内秒
        
        返回：
            dict or None - 星历字典，找不到返回None
        """
        if prn not in self.ephemerides:
            return None

        eph_list = self.ephemerides[prn]
        min_diff = float('inf')
        best_eph = None

        for eph in eph_list:
            # 计算周归化后的时间差
            diff = abs(self._week_wrap(time - eph['toe']))
            if diff < min_diff and diff < 7200:  # 星历有效期±2小时
                min_diff = diff
                best_eph = eph

        return best_eph

    def _ecef_to_geodetic(self, x, y, z):
        """
        ECEF坐标转换为大地坐标（经纬度和高度）
        
        使用迭代法求解大地纬度，收敛精度1e-12弧度
        
        参数：
            x, y, z: float - ECEF坐标（米）
        
        返回：
            tuple - (纬度rad, 经度rad, 高度m)
        """
        a = 6378137.0           # WGS-84椭球长半轴
        f = 1.0 / 298.257223563 # 扁率
        e2 = f * (2.0 - f)      # 第一偏心率平方

        r = math.sqrt(x**2 + y**2)  # 到Z轴的距离
        lon = math.atan2(y, x)      # 经度

        phi = math.atan2(z, r)  # 纬度初始值

        # 迭代求解大地纬度
        for _ in range(10):
            phi_prev = phi
            N = a / math.sqrt(1.0 - e2 * math.sin(phi)**2)  # 卯酉圈曲率半径
            h = r / math.cos(phi) - N                       # 高度
            phi = math.atan2(z / r, 1.0 - e2 * N / (N + h))

            if abs(phi - phi_prev) < 1e-12:
                break

        N = a / math.sqrt(1.0 - e2 * math.sin(phi)**2)
        h = r / math.cos(phi) - N
        lat = phi

        return lat, lon, h

    def _compute_satellite_position(self, eph, t):
        """
        计算卫星在ECEF坐标系中的位置
        
        使用广播星历参数计算卫星位置，包含摄动项修正
        算法流程：
        1. 计算平均角速度和平近点角
        2. 迭代求解偏近点角（牛顿法）
        3. 计算真近点角和纬度幅角
        4. 应用摄动项修正（delta-u, delta-r, delta-i）
        5. 计算轨道平面坐标
        6. 旋转到ECEF坐标系（考虑地球自转）
        
        参数：
            eph: dict - 卫星星历
            t: float - GPS周内秒（发射时刻）
        
        返回：
            numpy.array - [x, y, z] ECEF坐标（米）
        """
        # 计算相对于星历参考时刻的时间差（周归化）
        tk = self._week_wrap(t - eph['toe'])

        # 计算轨道半长轴和平均角速度
        A = eph['sqrt_a'] ** 2                  # 半长轴的平方 (m^2)
        n0 = math.sqrt(self.GM / (A ** 3))      # 平均角速度（无摄动） (rad/s)
        n = n0 + eph['delta_n']                 # 平均角速度（含摄动） (rad/s)
        M = eph['m0'] + n * tk                  # 平近点角 (rad) - 从参考时刻起算的角度

        # 迭代求解偏近点角 E（牛顿法）
        E = M
        for _ in range(30):
            E_new = M + eph['e'] * math.sin(E)
            if abs(E_new - E) < 1e-13:
                E = E_new
                break
            E = E_new

        # 计算真近点角和纬度幅角
        v = math.atan2(math.sqrt(1 - eph['e'] ** 2) * math.sin(E), math.cos(E) - eph['e'])
        phi = v + eph['w']  # 纬度幅角

        # 计算摄动项（使用sin(2phi)和cos(2phi)）
        sin_2phi = math.sin(2 * phi)
        cos_2phi = math.cos(2 * phi)

        du = eph['c_us'] * sin_2phi + eph['c_uc'] * cos_2phi  # 纬度幅角修正
        dr = eph['c_rs'] * sin_2phi + eph['c_rc'] * cos_2phi  # 轨道半径修正
        di = eph['c_is'] * sin_2phi + eph['c_ic'] * cos_2phi  # 轨道倾角修正

        # 修正后的轨道参数
        u = phi + du                                        # 修正后的纬度幅角 (rad)
        r = A * (1 - eph['e'] * math.cos(E)) + dr          # 修正后的轨道半径 (m) - 使用开普勒方程
        i = eph['i0'] + eph['idot'] * tk + di              # 修正后的轨道倾角 (rad) - 含倾角变化率

        # 轨道平面内坐标
        x_orb = r * math.cos(u)
        y_orb = r * math.sin(u)

        # 升交点赤经（考虑地球自转）
        Omega = eph['omega0'] + (eph['omegadot'] - self.omega_e) * tk - self.omega_e * eph['toe']

        # 转换到ECEF坐标系
        x = x_orb * math.cos(Omega) - y_orb * math.cos(i) * math.sin(Omega)
        y = x_orb * math.sin(Omega) + y_orb * math.cos(i) * math.cos(Omega)
        z = y_orb * math.sin(i)

        return np.array([x, y, z])

    def _compute_satellite_clock_error(self, eph, t):
        """
        计算卫星钟差
        
        钟差 = 多项式钟差 + 相对论改正 - TGD
        多项式钟差 = a0 + a1*tc + a2*tc^2
        相对论改正 = F * e * sqrt(a) * sin(E)
        
        参数：
            eph: dict - 卫星星历
            t: float - GPS周内秒
        
        返回：
            float - 卫星钟差（秒）
        """
        tc = self._week_wrap(t - eph['toc'])  # 相对于钟参考时刻的时间差
        tk = self._week_wrap(t - eph['toe'])  # 相对于轨道参考时刻的时间差
        
        # 计算偏近点角用于相对论改正
        A = eph['sqrt_a'] ** 2
        n0 = math.sqrt(self.GM / (A ** 3))
        n = n0 + eph['delta_n']
        M = eph['m0'] + n * tk
        
        E = M
        for _ in range(30):
            E_new = M + eph['e'] * math.sin(E)
            if abs(E_new - E) < 1e-13:
                E = E_new
                break
            E = E_new
        
        # 相对论改正
        F = -4.442807633e-10
        dtr = F * eph['e'] * eph['sqrt_a'] * math.sin(E)
        
        # 钟差 = 多项式 + 相对论改正 - TGD
        clock = eph['a0'] + eph['a1'] * tc + eph['a2'] * tc**2
        clock += dtr
        
        if 'tgd' in eph:
            clock -= eph['tgd']
        
        return clock

    def _earth_rotation_correction(self, sat_pos, rho):
        """
        地球自转改正
        
        由于信号传播时间内地球发生自转，需要对卫星位置进行修正
        
        参数：
            sat_pos: numpy.array - 卫星ECEF坐标
            rho: float - 卫星到接收机的距离（米）
        
        返回：
            numpy.array - 修正后的卫星ECEF坐标
        """
        tau = rho / self.c          # 信号传播时间
        theta = self.omega_e * tau  # 地球自转角位移

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        x = sat_pos[0]
        y = sat_pos[1]
        z = sat_pos[2]

        return np.array([
            cos_t * x + sin_t * y,
            -sin_t * x + cos_t * y,
            z
        ])

    def _compute_klobuchar(self, lat_rad, lon_rad, elevation_rad, azimuth_rad, sow):
        """
        使用Klobuchar模型计算电离层延迟
        
        Klobuchar模型是GPS广播星历提供的电离层延迟模型，适用于单频接收机。
        模型将电离层延迟分为昼夜两部分，白天使用余弦函数表示，夜间为常数。
        
        参数：
            lat_rad: float - 接收机纬度（弧度）
            lon_rad: float - 接收机经度（弧度）
            elevation_rad: float - 卫星高度角（弧度）
            azimuth_rad: float - 卫星方位角（弧度）
            sow: float - GPS周内秒
        
        返回：
            float - 电离层延迟（米）
        """
        if elevation_rad <= 0:
            return 0.0

        # 转换为半周期（0到1）
        lat = lat_rad / math.pi
        lon = lon_rad / math.pi
        el = elevation_rad / math.pi
        az = azimuth_rad

        # 计算倾斜因子和投影点坐标
        psi = 0.0137 / (el + 0.11) - 0.022  # 倾斜因子 - 信号路径与垂直方向的比例

        phi_i = lat + psi * math.cos(az)     # 电离层穿刺点纬度（归一化到-1~1）
        phi_i = max(min(phi_i, 0.416), -0.416)  # 限制在±23.8度范围内（电离层活跃区）

        lam_i = lon + psi * math.sin(az) / math.cos(phi_i * math.pi)  # 穿刺点经度（归一化）

        phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * math.pi)  # 修正纬度 - 考虑磁赤道偏移

        # 计算本地太阳时
        t = 43200.0 * lam_i + sow
        t = t % 86400.0

        # 计算振幅和周期（使用广播星历提供的alpha/beta系数）
        F = 1.0 + 16.0 * (0.53 - el) ** 3  # 高度角因子 - 低高度角时放大延迟

        # 振幅系数（s）- 由alpha参数多项式计算
        AMP = self.alpha[0] + self.alpha[1] * phi_m + self.alpha[2] * phi_m ** 2 + self.alpha[3] * phi_m ** 3
        if AMP < 0:
            AMP = 0  # 振幅不能为负

        # 周期系数（s）- 由beta参数多项式计算
        PER = self.beta[0] + self.beta[1] * phi_m + self.beta[2] * phi_m ** 2 + self.beta[3] * phi_m ** 3
        if PER < 72000:
            PER = 72000  # 周期最小值为20小时

        # 计算电离层延迟
        x = 2.0 * math.pi * (t - 50400.0) / PER  # 归一化时间参数（50400=14:00本地时）

        if abs(x) < 1.57:  # 白天（当地时间约07:00-19:00）- 余弦模型
            delay_sec = F * (5e-9 + AMP * (1 - x ** 2 / 2 + x ** 4 / 24))  # 泰勒展开余弦函数
        else:  # 夜间（约19:00-07:00）- 常数延迟
            delay_sec = F * 5e-9  # 夜间电离层延迟较小，约5ns

        return delay_sec * self.c  # 转换为距离（米）

    def _compute_troposphere(self, elevation):
        """
        计算对流层延迟（简化模型）
        
        使用Saastamoinen模型的简化形式，假设标准大气条件（气压1013.25 hPa）
        
        参数：
            elevation: float - 卫星高度角（弧度）
        
        返回：
            float - 对流层延迟（米）
        """
        if elevation <= math.radians(5):
            return 0.0
        return 2.3 / math.sin(elevation)

    def _compute_elevation_azimuth(self, sat_pos, rec_pos):
        """
        计算卫星相对于接收机的高度角和方位角
        
        参数：
            sat_pos: numpy.array - 卫星ECEF坐标
            rec_pos: numpy.array - 接收机ECEF坐标
        
        返回：
            tuple - (高度角rad, 方位角rad)
        """
        # 将接收机坐标转换为大地坐标
        lat, lon, _ = self._ecef_to_geodetic(rec_pos[0], rec_pos[1], rec_pos[2])

        # 计算卫星相对于接收机的向量
        dx = sat_pos[0] - rec_pos[0]
        dy = sat_pos[1] - rec_pos[1]
        dz = sat_pos[2] - rec_pos[2]

        # 转换到当地水平坐标系（东-北-天）
        E = -math.sin(lon) * dx + math.cos(lon) * dy  # 东向分量
        N = -math.sin(lat) * math.cos(lon) * dx - math.sin(lat) * math.sin(lon) * dy + math.cos(lat) * dz  # 北向分量
        U = math.cos(lat) * math.cos(lon) * dx + math.cos(lat) * math.sin(lon) * dy + math.sin(lat) * dz  # 天顶分量

        # 计算高度角和方位角
        el = math.atan2(U, math.sqrt(E**2 + N**2))
        az = math.atan2(E, N)

        if az < 0:
            az += 2 * math.pi

        return el, az

    def _solve_epoch_once(self, epoch_data, X, cdt_r, excluded_prns=None, debug=False):
        """
        单次历元解算（完整迭代收敛）
        
        执行加权最小二乘迭代求解，直到位置修正小于1e-4米或达到最大迭代次数
        
        参数：
            epoch_data: list - 历元卫星观测数据列表
            X: numpy.array - 接收机位置初值 [X, Y, Z]
            cdt_r: float - 接收机钟差初值（米）
            excluded_prns: set - 需要排除的卫星PRN集合
            debug: bool - 是否输出调试信息
        
        返回：
            tuple - (位置坐标, 钟差, 残差数组, 使用的PRN列表, 卫星详细信息)
        """
        if excluded_prns is None:
            excluded_prns = set()
        
        # 用于存储卫星详细信息（用于debug输出）
        sat_details = []
        
        # 迭代收敛
        for iter_num in range(20):
            A = []           # 设计矩阵
            L = []           # 观测值向量
            weights = []     # 权系数
            prn_list = []    # 使用的卫星PRN列表

            for sat in epoch_data:
                prn = sat['prn']
                P = sat['c1']  # C1C伪距
                
                # 跳过已排除的卫星
                if prn in excluded_prns:
                    continue
                
                # 只使用GPS卫星（PRN 1-32）
                if prn < 1 or prn > 32:
                    continue
                
                # 伪距范围检查（1e7 ~ 5e7米）
                if P < 1e7 or P > 5e7:
                    continue

                tr = sat['gpst']              # 接收时刻（GPS周内秒）
                ts_initial = tr - P / self.c  # 初步发射时刻

                # 查找星历
                eph = self._find_ephemeris(prn, ts_initial)
                if eph is None:
                    continue

                # 计算卫星钟差和修正发射时刻
                dts = self._compute_satellite_clock_error(eph, ts_initial)
                ts = tr - P / self.c - dts  # 修正后的发射时刻

                # 计算卫星位置
                sat_pos = self._compute_satellite_position(eph, ts)
                dts = self._compute_satellite_clock_error(eph, ts)

                # 计算几何距离并进行地球自转改正
                rho0 = np.linalg.norm(sat_pos - X)
                sat_pos = self._earth_rotation_correction(sat_pos, rho0)

                # 计算卫星到接收机的向量和距离
                vec = sat_pos - X
                rho = np.linalg.norm(vec)

                # 计算高度角和方位角，高度角低于10度的卫星不使用
                el, az = self._compute_elevation_azimuth(sat_pos, X)
                if el < math.radians(10):
                    continue

                # 计算电离层和对流层延迟
                lat, lon, h = self._ecef_to_geodetic(X[0], X[1], X[2])
                iono = self._compute_klobuchar(lat, lon, el, az, tr)
                trop = self._compute_troposphere(el)

                # 计算伪距观测方程的残差
                computed_pr = rho + cdt_r - self.c * dts + iono + trop
                L_i = P - computed_pr

                # 计算设计矩阵的一行（方向余弦）
                A_i = [
                    -(sat_pos[0] - X[0]) / rho,
                    -(sat_pos[1] - X[1]) / rho,
                    -(sat_pos[2] - X[2]) / rho,
                    1.0
                ]

                # 高度角相关权（sin(el)^2）
                weight_i = math.sin(el) ** 2

                # 收集卫星详细信息用于debug
                if self.debug or debug:
                    sat_details.append({
                        'prn': prn,
                        'P': P,
                        'tr': tr,
                        'ts': ts,
                        'toe': eph['toe'],
                        'dts_m': self.c * dts,  # 卫星钟差（米）
                        'sat_pos': sat_pos,
                        'el_deg': math.degrees(el),
                        'az_deg': math.degrees(az),
                        'iono': iono,
                        'trop': trop
                    })

                A.append(A_i)
                L.append(L_i)
                weights.append(weight_i)
                prn_list.append(prn)

            # 至少需要4颗卫星
            if len(A) < 4:
                return None, None, None, [], []

            # 转换为numpy数组并求解
            A = np.array(A)
            L = np.array(L)
            weights = np.array(weights)

            # 加权最小二乘
            W_sqrt = np.diag(np.sqrt(weights))
            dx = np.linalg.lstsq(W_sqrt @ A, W_sqrt @ L, rcond=None)[0]

            # 更新位置和钟差
            X += dx[:3]
            cdt_r += dx[3]

            # 收敛判断
            if np.linalg.norm(dx[:3]) < 1e-4:
                break

        # 计算最终残差
        residuals = None
        if len(A) >= 4:
            residuals = L - A @ dx
            # 将残差添加到卫星详细信息中
            for i, detail in enumerate(sat_details):
                detail['residual'] = residuals[i] if i < len(residuals) else 0.0

        return X, cdt_r, residuals, prn_list, sat_details

    def solve_epoch(self, epoch_data, initial_coords=None, debug=False):
        """
        求解单历元定位（带Klobuchar电离层改正）
        
        包含粗差剔除功能，最多剔除2颗残差最大的卫星
        
        参数：
            epoch_data: list - 历元卫星观测数据列表
            initial_coords: list - 初始坐标（可选）
            debug: bool - 是否输出调试信息
        
        返回：
            tuple - (坐标list, 钟差m, RMSm, 最大残差m, 使用卫星数, 失败原因, 调试信息)
        """
        # 设置初始坐标
        if initial_coords is None:
            if self.station_coords is not None:
                X = np.array(self.station_coords, dtype=np.float64)
            else:
                X = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        else:
            X = np.array(initial_coords, dtype=np.float64)

        cdt_r = 0.0  # 接收机钟差（距离单位，米）
        debug_info = []
        excluded_prns = set()
        max_exclusions = 2  # 最多剔除2颗卫星
        threshold = 50.0    # 残差阈值（米）
        final_sat_details = []

        # 粗差剔除循环
        for exclusion_round in range(max_exclusions + 1):
            # 完整迭代解算
            result_X, result_cdt_r, residuals, prn_list, sat_details = self._solve_epoch_once(
                epoch_data, X.copy(), cdt_r, excluded_prns, debug or self.debug
            )

            if result_X is None:
                break

            # 保存最后一次迭代的卫星详细信息
            final_sat_details = sat_details

            # 检查是否需要剔除
            if exclusion_round < max_exclusions and len(prn_list) >= 5:
                max_res_idx = np.argmax(np.abs(residuals))
                max_res = np.abs(residuals[max_res_idx])

                if max_res > threshold:
                    prn_to_exclude = prn_list[max_res_idx]
                    excluded_prns.add(prn_to_exclude)
                    if debug or self.debug:
                        debug_info.append(f"剔除卫星 PRN{prn_to_exclude:02d}，残差 {residuals[max_res_idx]:.2f} m")
                    # 更新下一轮的初值
                    X = result_X
                    cdt_r = result_cdt_r
                    continue

            # 不需要继续剔除，使用当前结果
            X = result_X
            cdt_r = result_cdt_r
            break

        # 计算最终 RMS 和最大残差
        rms = 0.0
        max_residual = 0.0
        if residuals is not None and len(residuals) > 0:
            rms = np.sqrt(np.mean(residuals ** 2))
            max_residual = np.max(np.abs(residuals))

        # Debug输出：每个卫星的详细信息
        if self.debug or debug:
            print("\n" + "="*80)
            print("卫星详细信息:")
            print("-"*80)
            print(f"{'PRN':>4} {'P (m)':>12} {'tr (s)':>12} {'ts (s)':>12} {'toe (s)':>12} {'ts-toe (s)':>12} "
                  f"{'dts (m)':>10} {'el (deg)':>10} {'az (deg)':>10} {'iono (m)':>10} {'trop (m)':>10} {'residual (m)':>12}")
            print("-"*80)
            for detail in final_sat_details:
                print(f"{detail['prn']:4d} {detail['P']:12.1f} {detail['tr']:12.3f} {detail['ts']:12.3f} "
                      f"{detail['toe']:12.3f} {(detail['ts'] - detail['toe']):12.3f} "
                      f"{detail['dts_m']:10.3f} {detail['el_deg']:10.1f} {detail['az_deg']:10.1f} "
                      f"{detail['iono']:10.3f} {detail['trop']:10.3f} {detail.get('residual', 0.0):12.3f}")
            print("-"*80)
            
            # 历元结束后的汇总信息
            print(f"\n使用卫星数: {len(prn_list)}")
            print(f"RMS: {rms:.3f} m")
            print(f"最大残差: {max_residual:.3f} m")
            print(f"解算坐标: X={X[0]:.4f} m, Y={X[1]:.4f} m, Z={X[2]:.4f} m")
            print(f"接收机钟差: {(cdt_r / self.c * 1e6):.2f} us")
            print("="*80)

        if debug or self.debug:
            if excluded_prns:
                debug_info.append(f"本历元共剔除 {len(excluded_prns)} 颗卫星: {sorted(excluded_prns)}")

        # 判断是否解算失败
        failure_reason = None
        if X is None:
            if len([s for s in epoch_data if 1 <= s['prn'] <= 32]) < 4:
                failure_reason = "有效卫星数不足（少于4颗）"
            else:
                failure_reason = "矩阵病态或其他解算错误"

        # 返回：坐标、钟差、RMS、最大残差、使用卫星数、失败原因、调试信息
        return X.tolist() if X is not None else None, cdt_r, rms, max_residual, len(prn_list), failure_reason, debug_info

    def process_file(self, max_epochs=100, debug=False):
        """
        处理观测文件并进行单点定位
        
        参数：
            max_epochs: int - 最大处理历元数（默认100）
            debug: bool - 是否输出调试信息
        
        返回：
            list - 定位结果列表，每个元素是包含历元信息的字典
        """
        results = []

        with open(self.obs_file, 'r') as f:
            lines = f.readlines()

        header = lines[0].strip().split(',')
        epoch_dict = {}

        # C1C伪距有效范围（米）
        MIN_PSEUDORANGE = 1e7
        MAX_PSEUDORANGE = 5e7

        for line in lines[1:]:
            parts = line.strip().split(',')
            epoch_time = parts[0]

            # === 卫星号处理：只使用GPS卫星 ===
            sat_id = parts[9].strip()
            
            # 如果卫星号是字符串格式（如G11、C03、E19等）
            if sat_id.isalpha() or (sat_id and not sat_id[0].isdigit()):
                # 只保留以G开头的GPS卫星
                if not sat_id.startswith('G'):
                    continue  # 跳过非GPS卫星（如C-北斗、E-伽利略、R-GLONASS、J-准天顶）
                # 解析PRN：Gxx -> xx
                try:
                    prn = int(sat_id[1:])
                except:
                    continue
            else:
                # 卫星号已经是数字格式
                try:
                    prn = int(sat_id)
                except:
                    continue
            
            # GPS卫星PRN范围：1-32
            if prn < 1 or prn > 32:
                continue

            # === C1C伪距处理 ===
            try:
                c1 = float(parts[10])
            except:
                continue
            
            # C1C伪距范围检查（1e7 ~ 5e7 米）
            if c1 < MIN_PSEUDORANGE or c1 > MAX_PSEUDORANGE:
                if debug or self.debug:
                    print(f"跳过 PRN{prn:02d}: C1C伪距 {c1:.1f}m 超出有效范围")
                continue

            if epoch_time not in epoch_dict:
                epoch_dict[epoch_time] = []

            # 只使用GPS卫星的C1C伪距进行单频SPP
            sat_data = {
                'prn': prn,
                'c1': c1,  # C1C伪距（米）
                'year': int(parts[1]),
                'month': int(parts[2]),
                'day': int(parts[3]),
                'hour': int(parts[4]),
                'minute': int(parts[5]),
                'second': float(parts[6])
            }

            # === 将日历时间转换为GPS周内秒 ===
            sat_data['gpst'] = self._compute_gpst(
                sat_data['year'], sat_data['month'], sat_data['day'],
                sat_data['hour'], sat_data['minute'], sat_data['second'])

            epoch_dict[epoch_time].append(sat_data)

        # 保存上一历元的坐标作为下一历元的初值
        prev_coords = self.station_coords

        for i, (epoch_time, sats) in enumerate(epoch_dict.items()):
            if i >= max_epochs:
                break

            # 调用solve_epoch，获取：坐标、钟差、RMS、最大残差、使用卫星数、失败原因、调试信息
            X, dt, rms, max_residual, num_sats_used, failure_reason, debug_info = self.solve_epoch(
                sats, initial_coords=prev_coords, debug=debug
            )

            if debug and debug_info:
                print(f"\n=== {epoch_time} ===")
                for info in debug_info:
                    print(info)

            if X is not None:
                results.append({
                    'epoch': epoch_time,
                    'x': X[0], 'y': X[1], 'z': X[2],
                    'clock_offset': dt,
                    'num_satellites': num_sats_used,  # 使用的卫星数（剔除粗差后）
                    'rms': rms,
                    'max_residual': max_residual,
                    'failure_reason': None
                })
                prev_coords = X
            else:
                # 解算失败，记录失败原因
                results.append({
                    'epoch': epoch_time,
                    'x': None, 'y': None, 'z': None,
                    'clock_offset': None,
                    'num_satellites': num_sats_used,
                    'rms': None,
                    'max_residual': None,
                    'failure_reason': failure_reason
                })

        return results

def main():
    """
    主函数：演示单点定位算法的使用
    
    读取观测数据和星历文件，进行单点定位，输出结果到文件
    """
    # 输入文件路径
    obs_file = r'd:\卫星导航算法大作业\CPP-1\Data_01\obs_data\DAV100ATA_R_20241290000_01D_30S_MO_obs.csv'
    ephemeris_file = r'd:\卫星导航算法大作业\CPP-1\Data_01\ephemeris\brdc1290_ephemerides.csv'
    klobuchar_file = r'd:\卫星导航算法大作业\CPP-1\Data_01\ephemeris\brdc1290_klobuchar.csv'

    # 测站近似坐标（ITRF XYZ，单位：米）
    approx_coords = [486854.546000, 2285099.292400, -5914955.713600]

    # 创建求解器并处理数据
    solver = SPPSolver(obs_file, ephemeris_file, klobuchar_file, approx_coords)
    results = solver.process_file(max_epochs=100, debug=False)

    # 输出结果到文件
    output_file = r'd:\卫星导航算法大作业\CPP-1\spp_results_klobuchar.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("GPS Single Point Positioning Results (with Klobuchar Ionosphere Model)\n")
        f.write("="*130 + "\n")
        f.write(f"Station: DAV100ATA\n")
        f.write(f"Approx Position: X={approx_coords[0]:.4f}, Y={approx_coords[1]:.4f}, Z={approx_coords[2]:.4f}\n")
        f.write(f"GPS Week: {solver.gps_week}\n")
        f.write("\n")
        # 输出表头
        f.write(f"{'Epoch':<23} {'X (m)':>15} {'Y (m)':>15} {'Z (m)':>15} {'dX (m)':>10} {'dY (m)':>10} {'dZ (m)':>10} {'Clock (us)':>12} {'Satellites':>10} {'RMS (m)':>10} {'Max Res (m)':>12} {'Status':<30}\n")
        f.write("-"*130 + "\n")

        success_count = 0
        fail_count = 0

        for res in results:
            if res['failure_reason'] is not None:
                # 解算失败
                f.write(f"{res['epoch']:<23} {'-':>15} {'-':>15} {'-':>15} {'-':>10} {'-':>10} {'-':>10} {'-':>12} {res['num_satellites']:>10} {'-':>10} {'-':>12} {res['failure_reason']:<30}\n")
                fail_count += 1
            else:
                # 解算成功
                clock_us = (res['clock_offset'] / solver.c) * 1e6
                dx = res['x'] - approx_coords[0]
                dy = res['y'] - approx_coords[1]
                dz = res['z'] - approx_coords[2]
                rms = res['rms'] if res['rms'] is not None else 0.0
                max_res = res['max_residual'] if res['max_residual'] is not None else 0.0
                f.write(f"{res['epoch']:<23} {res['x']:>15.4f} {res['y']:>15.4f} {res['z']:>15.4f} {dx:>10.4f} {dy:>10.4f} {dz:>10.4f} {clock_us:>12.2f} {res['num_satellites']:>10} {rms:>10.4f} {max_res:>12.4f} {'Success':<30}\n")
                success_count += 1

        f.write("-"*130 + "\n")
        f.write(f"Summary: {success_count} successful epochs, {fail_count} failed epochs\n")

    print(f"定位结果已保存到: {output_file}")
    print(f"共处理 {len(results)} 个历元 ({success_count} 成功, {fail_count} 失败)")

if __name__ == '__main__':
    main()
