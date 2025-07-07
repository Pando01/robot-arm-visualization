"""
Human-Robot Trajectory Control Simulation Core
핵심 시뮬레이션 로직을 담당하는 모듈
"""

import numpy as np
from collections import deque
from scipy.integrate import simpson

class HumanMotionSensor:
    """인간 동작 센서 시뮬레이터"""
    
    def __init__(self):
        self.is_running = False
        self.joint_angles = {
            'shoulder': 0.0,
            'elbow': 0.0,
            'wrist': 0.0
        }
        self.noise_level = 0.05  # 센서 노이즈 레벨
        
    def simulate_human_motion(self, t):
        """사람의 자연스러운 팔 움직임 시뮬레이션"""
        # 어깨: 천천히 위아래 움직임
        shoulder = 30 * np.sin(0.5 * t) + 15 * np.sin(0.3 * t)
        
        # 팔꿈치: 주기적인 굽힘/펴짐
        elbow = 45 + 30 * np.sin(0.8 * t) + 10 * np.cos(0.6 * t)
        
        # 손목: 작은 회전 움직임
        wrist = 20 * np.sin(1.2 * t) + 5 * np.cos(0.9 * t)
        
        # 노이즈 추가 (실제 센서의 불완전함 시뮬레이션)
        shoulder += np.random.normal(0, self.noise_level * abs(shoulder))
        elbow += np.random.normal(0, self.noise_level * abs(elbow))
        wrist += np.random.normal(0, self.noise_level * abs(wrist))
        
        return shoulder, elbow, wrist
    
    def get_current_angles(self, t):
        """현재 시간의 관절 각도 반환"""
        shoulder, elbow, wrist = self.simulate_human_motion(t)
        
        self.joint_angles = {
            'shoulder': shoulder,
            'elbow': elbow,
            'wrist': wrist
        }
        
        return self.joint_angles

class TrigonometricSpline:
    """삼각함수 기반 스플라인 곡선 생성기"""
    
    def __init__(self, n_harmonics=3):
        self.n_harmonics = n_harmonics  # 조화파 개수
        self.coefficients = {}
        
    def fit(self, t_data, angle_data, joint_name):
        """시간-각도 데이터에 삼각함수 스플라인 피팅"""
        if len(t_data) < 2:
            return
            
        # 기본 주파수 계산 (데이터 길이 기반)
        T = t_data[-1] - t_data[0] if len(t_data) > 1 else 1
        omega = 2 * np.pi / max(T, 1)
        
        # 삼각함수 기저 행렬 생성
        A = np.ones((len(t_data), 1 + 2 * self.n_harmonics))
        
        # 상수항
        A[:, 0] = 1
        
        # 조화파 항들
        for k in range(1, self.n_harmonics + 1):
            A[:, 2*k-1] = np.cos(k * omega * t_data)
            A[:, 2*k] = np.sin(k * omega * t_data)
        
        # 최소제곱법으로 계수 계산
        try:
            coeffs = np.linalg.lstsq(A, angle_data, rcond=None)[0]
            self.coefficients[joint_name] = {
                'coeffs': coeffs,
                'omega': omega
            }
        except np.linalg.LinAlgError:
            # 계산 실패 시 기본값 사용
            self.coefficients[joint_name] = {
                'coeffs': np.zeros(1 + 2 * self.n_harmonics),
                'omega': omega
            }
    
    def predict(self, t, joint_name):
        """주어진 시간에서의 각도 예측"""
        if joint_name not in self.coefficients:
            return 0.0
            
        coeffs = self.coefficients[joint_name]['coeffs']
        omega = self.coefficients[joint_name]['omega']
        
        # 삼각함수 스플라인 계산
        result = coeffs[0]  # 상수항
        
        for k in range(1, self.n_harmonics + 1):
            if 2*k < len(coeffs):
                result += coeffs[2*k-1] * np.cos(k * omega * t)
                result += coeffs[2*k] * np.sin(k * omega * t)
        
        return result
    
    def predict_velocity(self, t, joint_name):
        """각속도 계산 (1차 도함수)"""
        if joint_name not in self.coefficients:
            return 0.0
            
        coeffs = self.coefficients[joint_name]['coeffs']
        omega = self.coefficients[joint_name]['omega']
        
        result = 0.0
        for k in range(1, self.n_harmonics + 1):
            if 2*k < len(coeffs):
                result += -k * omega * coeffs[2*k-1] * np.sin(k * omega * t)
                result += k * omega * coeffs[2*k] * np.cos(k * omega * t)
        
        return result
    
    def predict_acceleration(self, t, joint_name):
        """각가속도 계산 (2차 도함수)"""
        if joint_name not in self.coefficients:
            return 0.0
            
        coeffs = self.coefficients[joint_name]['coeffs']
        omega = self.coefficients[joint_name]['omega']
        
        result = 0.0
        for k in range(1, self.n_harmonics + 1):
            if 2*k < len(coeffs):
                result += -k**2 * omega**2 * coeffs[2*k-1] * np.cos(k * omega * t)
                result += -k**2 * omega**2 * coeffs[2*k] * np.sin(k * omega * t)
        
        return result

class RobotTrajectoryController:
    """로봇 궤적 제어기"""
    
    def __init__(self):
        self.joint_limits = {
            'shoulder': (-90, 90),
            'elbow': (0, 150),
            'wrist': (-60, 60)
        }
        self.velocity_limits = {
            'shoulder': 50,  # deg/s
            'elbow': 80,
            'wrist': 100
        }
        self.scaling_factors = {
            'shoulder': 0.8,  # 인간-로봇 크기 차이 보정
            'elbow': 0.9,
            'wrist': 1.0
        }
        
    def human_to_robot_mapping(self, human_angles):
        """인간 관절 각도를 로봇 관절 각도로 변환"""
        robot_angles = {}
        
        for joint, angle in human_angles.items():
            # 스케일링 적용
            scaled_angle = angle * self.scaling_factors[joint]
            
            # 관절 한계 적용
            min_limit, max_limit = self.joint_limits[joint]
            robot_angles[joint] = np.clip(scaled_angle, min_limit, max_limit)
            
        return robot_angles
    
    def velocity_limiting(self, current_angles, target_angles, dt=0.01):
        """속도 제한 적용"""
        limited_angles = {}
        
        for joint in target_angles:
            if joint in current_angles:
                angle_diff = target_angles[joint] - current_angles[joint]
                max_change = self.velocity_limits[joint] * dt
                
                if abs(angle_diff) > max_change:
                    limited_change = np.sign(angle_diff) * max_change
                    limited_angles[joint] = current_angles[joint] + limited_change
                else:
                    limited_angles[joint] = target_angles[joint]
            else:
                limited_angles[joint] = target_angles[joint]
                
        return limited_angles

class RealTimeSimulation:
    """실시간 시뮬레이션 시스템"""
    
    def __init__(self, window_size=50):
        self.sensor = HumanMotionSensor()
        self.spline = TrigonometricSpline(n_harmonics=3)
        self.controller = RobotTrajectoryController()
        
        self.window_size = window_size
        self.time_window = deque(maxlen=window_size)
        self.human_data = {
            'shoulder': deque(maxlen=window_size),
            'elbow': deque(maxlen=window_size),
            'wrist': deque(maxlen=window_size)
        }
        # 스플라인 적용된 로봇 데이터
        self.robot_data_spline = {
            'shoulder': deque(maxlen=window_size),
            'elbow': deque(maxlen=window_size),
            'wrist': deque(maxlen=window_size)
        }
        # 스플라인 없는 로봇 데이터 (직접 매핑)
        self.robot_data_direct = {
            'shoulder': deque(maxlen=window_size),
            'elbow': deque(maxlen=window_size),
            'wrist': deque(maxlen=window_size)
        }
        
        self.current_time = 0
        self.dt = 0.05  # 50ms 업데이트 주기
        self.is_running = False
        
        self.current_robot_angles_spline = {'shoulder': 0, 'elbow': 0, 'wrist': 0}
        self.current_robot_angles_direct = {'shoulder': 0, 'elbow': 0, 'wrist': 0}
        
    def update_data(self):
        """데이터 업데이트 (센서 읽기 시뮬레이션)"""
        # 인간 동작 데이터 수집
        human_angles = self.sensor.get_current_angles(self.current_time)
        
        # 시간 창에 데이터 추가
        self.time_window.append(self.current_time)
        for joint, angle in human_angles.items():
            self.human_data[joint].append(angle)
        
        # 방법 1: 직접 매핑 (스플라인 없음)
        target_robot_angles_direct = self.controller.human_to_robot_mapping(human_angles)
        self.current_robot_angles_direct = self.controller.velocity_limiting(
            self.current_robot_angles_direct, target_robot_angles_direct, self.dt
        )
        
        for joint, angle in self.current_robot_angles_direct.items():
            self.robot_data_direct[joint].append(angle)
        
        # 방법 2: 스플라인 적용
        if len(self.time_window) >= 10:
            # 스플라인 피팅
            t_array = np.array(self.time_window)
            for joint in human_angles:
                angle_array = np.array(self.human_data[joint])
                self.spline.fit(t_array, angle_array, joint)
            
            # 미래 시점 예측
            future_time = self.current_time + self.dt
            predicted_angles = {}
            
            for joint in human_angles:
                predicted_angles[joint] = self.spline.predict(future_time, joint)
            
            # 인간-로봇 변환
            target_robot_angles_spline = self.controller.human_to_robot_mapping(predicted_angles)
            
            # 속도 제한 적용
            self.current_robot_angles_spline = self.controller.velocity_limiting(
                self.current_robot_angles_spline, target_robot_angles_spline, self.dt
            )
            
            # 스플라인 로봇 데이터 저장
            for joint, angle in self.current_robot_angles_spline.items():
                self.robot_data_spline[joint].append(angle)
        else:
            # 초기 데이터 부족 시 기본값
            for joint in human_angles:
                self.robot_data_spline[joint].append(0)
        
        self.current_time += self.dt
    
    def get_performance_metrics(self):
        """성능 지표 계산"""
        if len(self.time_window) < 10:
            return {}
        
        metrics = {'spline': {}, 'direct': {}}
        
        for joint in ['shoulder', 'elbow', 'wrist']:
            if (len(self.human_data[joint]) >= 10 and 
                len(self.robot_data_spline[joint]) >= 10 and
                len(self.robot_data_direct[joint]) >= 10):
                
                human_array = np.array(list(self.human_data[joint]))
                robot_spline_array = np.array(list(self.robot_data_spline[joint]))
                robot_direct_array = np.array(list(self.robot_data_direct[joint]))
                
                # 지연을 고려한 정렬 (최근 데이터 비교)
                min_len = min(len(human_array), len(robot_spline_array), len(robot_direct_array))
                if min_len > 5:
                    h_data = human_array[-min_len:]
                    r_spline_data = robot_spline_array[-min_len:]
                    r_direct_data = robot_direct_array[-min_len:]
                    
                    # 스플라인 적용 버전 성능
                    rmse_spline = np.sqrt(np.mean((h_data - r_spline_data)**2))
                    correlation_spline = np.correlate(h_data, r_spline_data, mode='full')
                    lag_spline = np.argmax(correlation_spline) - (len(r_spline_data) - 1)
                    delay_ms_spline = abs(lag_spline) * self.dt * 1000
                    
                    # 직접 매핑 버전 성능
                    rmse_direct = np.sqrt(np.mean((h_data - r_direct_data)**2))
                    correlation_direct = np.correlate(h_data, r_direct_data, mode='full')
                    lag_direct = np.argmax(correlation_direct) - (len(r_direct_data) - 1)
                    delay_ms_direct = abs(lag_direct) * self.dt * 1000
                    
                    # 부드러움 측정 (저크 계산)
                    if min_len > 2:
                        # 2차 차분으로 저크 근사
                        jerk_spline = np.mean(np.abs(np.diff(r_spline_data, n=2)))
                        jerk_direct = np.mean(np.abs(np.diff(r_direct_data, n=2)))
                    else:
                        jerk_spline = jerk_direct = 0
                    
                    metrics['spline'][joint] = {
                        'rmse': rmse_spline,
                        'delay_ms': delay_ms_spline,
                        'jerk': jerk_spline
                    }
                    
                    metrics['direct'][joint] = {
                        'rmse': rmse_direct,
                        'delay_ms': delay_ms_direct,
                        'jerk': jerk_direct
                    }
        
        return metrics
    
    def get_robot_arm_position(self, method='spline'):
        """로봇 팔의 현재 위치 계산"""
        if method == 'spline':
            data = self.robot_data_spline
        else:
            data = self.robot_data_direct
            
        if len(data['shoulder']) == 0:
            return None
            
        shoulder_angle = np.radians(data['shoulder'][-1])
        elbow_angle = np.radians(data['elbow'][-1])
        wrist_angle = np.radians(data['wrist'][-1])
        
        # 로봇 팔 세그먼트 계산
        L1, L2, L3 = 1.0, 0.8, 0.3  # 링크 길이
        
        # 어깨
        x1 = L1 * np.cos(shoulder_angle)
        y1 = L1 * np.sin(shoulder_angle)
        
        # 팔꿈치
        x2 = x1 + L2 * np.cos(shoulder_angle + elbow_angle)
        y2 = y1 + L2 * np.sin(shoulder_angle + elbow_angle)
        
        # 손목
        x3 = x2 + L3 * np.cos(shoulder_angle + elbow_angle + wrist_angle)
        y3 = y2 + L3 * np.sin(shoulder_angle + elbow_angle + wrist_angle)
        
        return {
            'shoulder': (0, 0),
            'elbow': (x1, y1),
            'wrist': (x2, y2),
            'end': (x3, y3)
        }