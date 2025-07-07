"""
Human-Robot Trajectory Control Visualization Module
시각화 및 사용자 인터페이스를 담당하는 모듈
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from collections import deque  # deque import 추가

class SimulationVisualizer:
    """시뮬레이션 시각화 클래스"""
    
    def __init__(self, simulation):
        self.simulation = simulation
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.bars = {}
        self.widgets = {}
        self.performance_text = None
        self.trajectory_data = {}  # 궤적 데이터용
        
        self.joint_names = ['Shoulder', 'Elbow', 'Wrist']
        self.joint_keys = ['shoulder', 'elbow', 'wrist']
        
        self.setup_figure()
        self.setup_plots()
        self.setup_controls()
        
    def setup_figure(self):
        """그래프 레이아웃 설정"""
        # 1920x1280 해상도에 최적화된 크기 설정 - 3x3 레이아웃으로 변경
        self.fig = plt.figure(figsize=(24, 16))
        
        # 서브플롯 간격 조정으로 텍스트 겹침 방지
        plt.subplots_adjust(
            left=0.05,      # 왼쪽 여백
            bottom=0.15,    # 아래쪽 여백 증가 (컨트롤 패널 공간)
            right=0.98,     # 오른쪽 여백
            top=0.92,       # 위쪽 여백 (제목 공간)
            wspace=0.2,     # 가로 간격
            hspace=0.45     # 세로 간격 증가 (텍스트 겹침 방지)
        )
        
        # 상단: 관절별 비교 (3개)
        self.axes['joint1'] = plt.subplot(3, 3, 1)  # Shoulder
        self.axes['joint2'] = plt.subplot(3, 3, 2)  # Elbow  
        self.axes['joint3'] = plt.subplot(3, 3, 3)  # Wrist
        
        # 중단: 로봇 팔 비교 (2개)
        self.axes['robot_spline'] = plt.subplot(3, 3, 4)   # Robot Arm Spline
        self.axes['robot_direct'] = plt.subplot(3, 3, 5)   # Robot Arm Direct
        self.axes['robot_overlay'] = plt.subplot(3, 3, 6)  # Overlay Comparison
        
        # 하단: 성능 비교 (3개)
        self.axes['performance'] = plt.subplot(3, 3, 7)    # RMSE Comparison
        self.axes['smoothness'] = plt.subplot(3, 3, 8)     # Jerk Comparison
        self.axes['trajectory'] = plt.subplot(3, 3, 9)     # End-effector Trajectory
        
        self.fig.suptitle('Comprehensive Spline vs Direct Mapping Comparison - Real-time Human-Robot Control', 
                         fontsize=13, fontweight='bold', y=0.98)
        
    def setup_plots(self):
        """각 플롯 설정"""
        # 개별 관절 비교 차트 설정
        joint_axes = [self.axes['joint1'], self.axes['joint2'], self.axes['joint3']]
        
        for i, (ax, name, key) in enumerate(zip(joint_axes, self.joint_names, self.joint_keys)):
            ax.set_title(f'{name} Joint Comparison', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Time (sec)', fontsize=12, labelpad=8)  # labelpad 추가
            ax.set_ylabel('Angle (deg)', fontsize=12, labelpad=8)  # labelpad 추가
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=10)
            
            # 초기 축 범위 설정 - 항상 0부터 시작
            ax.set_xlim(0, 10)  # 처음 10초 표시
            ax.set_ylim(-100, 100)  # 초기 Y축 범위
            
            # 라인 초기화
            self.lines[f'human_{key}'], = ax.plot([], [], 'b-', alpha=0.8, linewidth=2.5, label='Human')
            self.lines[f'robot_spline_{key}'], = ax.plot([], [], 'g-', alpha=0.8, linewidth=2.5, label='Robot-Spline')
            self.lines[f'robot_direct_{key}'], = ax.plot([], [], 'r--', alpha=0.8, linewidth=2.5, label='Robot-Direct')
            ax.legend(fontsize=10, loc='upper left', framealpha=0.9)  # 위치를 upper left로 변경
        
        # 성능 비교 차트
        ax_perf = self.axes['performance']
        ax_perf.set_title('RMSE Comparison', fontsize=14, fontweight='bold', pad=15)
        ax_perf.set_xlabel('Joint', fontsize=12, labelpad=8)
        ax_perf.set_ylabel('RMSE (deg)', fontsize=12, labelpad=8)
        ax_perf.grid(True, alpha=0.3)
        ax_perf.tick_params(labelsize=10)
        
        x_pos = np.arange(len(self.joint_names))
        width = 0.35
        self.bars['performance_spline'] = ax_perf.bar(x_pos - width/2, [0, 0, 0], width, 
                                                     label='Spline', color='green', alpha=0.7)
        self.bars['performance_direct'] = ax_perf.bar(x_pos + width/2, [0, 0, 0], width, 
                                                     label='Direct', color='red', alpha=0.7)
        ax_perf.set_xticks(x_pos)
        ax_perf.set_xticklabels(self.joint_names)
        ax_perf.legend(fontsize=11, loc='upper left', framealpha=0.9)  # 위치 변경
        
        # 로봇 팔 시각화 - 스플라인 방법
        ax_robot_spline = self.axes['robot_spline']
        ax_robot_spline.set_title('Robot Arm - Spline Method', fontsize=14, fontweight='bold', pad=15)
        ax_robot_spline.set_xlim(-3.2, 4.2)
        ax_robot_spline.set_ylim(-2.2, 2.2)
        ax_robot_spline.set_aspect('equal')
        ax_robot_spline.grid(True, alpha=0.3)
        ax_robot_spline.tick_params(labelsize=10)
        
        self.lines['robot_spline_upper'], = ax_robot_spline.plot([0, 0], [0, 0], 'g-', linewidth=6, label='Upper Arm')
        self.lines['robot_spline_forearm'], = ax_robot_spline.plot([0, 0], [0, 0], 'b-', linewidth=6, label='Forearm')
        self.lines['robot_spline_hand'], = ax_robot_spline.plot([0, 0], [0, 0], 'r-', linewidth=4, label='Hand')
        # 관절 포인트 추가
        self.lines['robot_spline_joints'], = ax_robot_spline.plot([], [], 'ko', markersize=8, label='Joints')
        ax_robot_spline.legend(fontsize=10, loc='upper left', framealpha=0.9)  # 위치 변경
        
        # 로봇 팔 시각화 - 직접 매핑 방법
        ax_robot_direct = self.axes['robot_direct']
        ax_robot_direct.set_title('Robot Arm - Direct Method', fontsize=14, fontweight='bold', pad=15)
        ax_robot_direct.set_xlim(-3.2, 4.2)
        ax_robot_direct.set_ylim(-2.2, 2.2)
        ax_robot_direct.set_aspect('equal')
        ax_robot_direct.grid(True, alpha=0.3)
        ax_robot_direct.tick_params(labelsize=10)
        
        self.lines['robot_direct_upper'], = ax_robot_direct.plot([0, 0], [0, 0], 'orange', linewidth=6, label='Upper Arm')
        self.lines['robot_direct_forearm'], = ax_robot_direct.plot([0, 0], [0, 0], 'purple', linewidth=6, label='Forearm')
        self.lines['robot_direct_hand'], = ax_robot_direct.plot([0, 0], [0, 0], 'brown', linewidth=4, label='Hand')
        # 관절 포인트 추가
        self.lines['robot_direct_joints'], = ax_robot_direct.plot([], [], 'ko', markersize=8, label='Joints')
        ax_robot_direct.legend(fontsize=10, loc='upper left', framealpha=0.9)  # 위치 변경
        
        # 로봇 팔 오버레이 비교
        ax_robot_overlay = self.axes['robot_overlay']
        ax_robot_overlay.set_title('Robot Arm Overlay Comparison', fontsize=14, fontweight='bold', pad=15)
        ax_robot_overlay.set_xlim(-3.2, 4.2)
        ax_robot_overlay.set_ylim(-2.2, 2.2)
        ax_robot_overlay.set_aspect('equal')
        ax_robot_overlay.grid(True, alpha=0.3)
        ax_robot_overlay.tick_params(labelsize=10)
        
        # 스플라인 (투명하게)
        self.lines['overlay_spline_upper'], = ax_robot_overlay.plot([0, 0], [0, 0], 'g-', linewidth=4, alpha=0.7, label='Spline Method')
        self.lines['overlay_spline_forearm'], = ax_robot_overlay.plot([0, 0], [0, 0], 'g-', linewidth=4, alpha=0.7)
        self.lines['overlay_spline_hand'], = ax_robot_overlay.plot([0, 0], [0, 0], 'g-', linewidth=3, alpha=0.7)
        
        # 직접 매핑 (점선으로)
        self.lines['overlay_direct_upper'], = ax_robot_overlay.plot([0, 0], [0, 0], 'r--', linewidth=4, alpha=0.7, label='Direct Method')
        self.lines['overlay_direct_forearm'], = ax_robot_overlay.plot([0, 0], [0, 0], 'r--', linewidth=4, alpha=0.7)
        self.lines['overlay_direct_hand'], = ax_robot_overlay.plot([0, 0], [0, 0], 'r--', linewidth=3, alpha=0.7)
        
        ax_robot_overlay.legend(fontsize=10, loc='upper left', framealpha=0.9)  # 위치 변경
        
        # 엔드 이펙터 궤적 비교
        ax_trajectory = self.axes['trajectory']
        ax_trajectory.set_title('End-Effector Trajectory Comparison', fontsize=14, fontweight='bold', pad=15)
        ax_trajectory.set_xlim(-0, 7)
        ax_trajectory.set_ylim(-2.2, 2.2)
        ax_trajectory.set_aspect('equal')
        ax_trajectory.grid(True, alpha=0.3)
        ax_trajectory.tick_params(labelsize=10)
        
        # 궤적 라인 초기화 (최근 N개 포인트의 궤적 표시)
        self.lines['trajectory_spline'], = ax_trajectory.plot([], [], 'g-', linewidth=2, alpha=0.8, label='Spline Trajectory')
        self.lines['trajectory_direct'], = ax_trajectory.plot([], [], 'r--', linewidth=2, alpha=0.8, label='Direct Trajectory')
        self.lines['trajectory_spline_current'], = ax_trajectory.plot([], [], 'go', markersize=8, label='Spline Current')
        self.lines['trajectory_direct_current'], = ax_trajectory.plot([], [], 'ro', markersize=8, label='Direct Current')
        
        ax_trajectory.legend(fontsize=10, loc='upper right', framealpha=0.9)
        
        # 궤적 데이터 저장용 deque 초기화
        self.trajectory_data = {
            'spline_x': deque(maxlen=100),
            'spline_y': deque(maxlen=100),
            'direct_x': deque(maxlen=100),
            'direct_y': deque(maxlen=100)
        }
        
        # 부드러움 비교 (저크)
        ax_smooth = self.axes['smoothness']
        ax_smooth.set_title('Smoothness Comparison (Jerk)', fontsize=14, fontweight='bold', pad=15)
        ax_smooth.set_xlabel('Joint', fontsize=12)
        ax_smooth.set_ylabel('Average Jerk', fontsize=12)
        ax_smooth.grid(True, alpha=0.3)
        ax_smooth.tick_params(labelsize=10)
        
        self.bars['jerk_spline'] = ax_smooth.bar(x_pos - width/2, [0, 0, 0], width, 
                                               label='Spline', color='green', alpha=0.7)
        self.bars['jerk_direct'] = ax_smooth.bar(x_pos + width/2, [0, 0, 0], width, 
                                               label='Direct', color='red', alpha=0.7)
        ax_smooth.set_xticks(x_pos)
        ax_smooth.set_xticklabels(self.joint_names)
        ax_smooth.legend(fontsize=11, framealpha=0.9)
        
    def setup_controls(self):
        """컨트롤 패널 설정"""
        # 노이즈 레벨 슬라이더 - 더 아래쪽으로 이동
        ax_noise = plt.axes([0.08, 0.06, 0.25, 0.03])
        self.widgets['noise_slider'] = Slider(ax_noise, 'Noise Level', 0.0, 0.2, 
                                             valinit=0.05, valfmt='%.3f')
        
        # 시작/정지 버튼 - 더 아래쪽으로 이동
        ax_button = plt.axes([0.38, 0.06, 0.12, 0.04])
        self.widgets['start_button'] = Button(ax_button, 'Start/Stop')
        
        # 성능 지표 텍스트 상자 - 더 아래쪽으로 이동
        ax_text = plt.axes([0.55, 0.02, 0.42, 0.09])
        ax_text.axis('off')
        
        # 배경 상자 추가로 가독성 향상
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8)
        self.performance_text = ax_text.text(0.02, 0.5, 
                                           'Performance Metrics will appear here...\n'
                                           'Green: Spline method (smoother)\n'
                                           'Red: Direct method (faster response)', 
                                           transform=ax_text.transAxes, 
                                           fontsize=11,
                                           verticalalignment='center',
                                           bbox=bbox_props)
        
        # 이벤트 핸들러 연결
        self.widgets['noise_slider'].on_changed(self.update_noise)
        self.widgets['start_button'].on_clicked(self.toggle_simulation)
        
    def update_noise(self, val):
        """노이즈 레벨 업데이트"""
        self.simulation.sensor.noise_level = val
        
    def toggle_simulation(self, event):
        """시뮬레이션 시작/정지"""
        self.simulation.is_running = not self.simulation.is_running
        
    def update_joint_plots(self):
        """관절별 플롯 업데이트"""
        if len(self.simulation.time_window) <= 1:
            return
            
        t_data = list(self.simulation.time_window)
        joint_axes = [self.axes['joint1'], self.axes['joint2'], self.axes['joint3']]
        
        # 개별 관절 차트 업데이트
        for i, key in enumerate(self.joint_keys):
            if len(self.simulation.human_data[key]) > 0:
                self.lines[f'human_{key}'].set_data(t_data, list(self.simulation.human_data[key]))
                self.lines[f'robot_spline_{key}'].set_data(t_data, list(self.simulation.robot_data_spline[key]))
                self.lines[f'robot_direct_{key}'].set_data(t_data, list(self.simulation.robot_data_direct[key]))
            
            # 축 범위 설정 개선
            if len(t_data) > 0:
                current_time = t_data[-1]
                
                # X축 범위: 항상 0부터 시작하도록 설정
                if current_time <= 10:
                    # 처음 10초는 0부터 10까지 고정
                    joint_axes[i].set_xlim(0, 10)
                else:
                    # 10초 이후부터는 슬라이딩 윈도우
                    joint_axes[i].set_xlim(current_time - 10, current_time + 1)
                
                # Y축 범위를 데이터에 맞게 조정
                if len(self.simulation.human_data[key]) > 0:
                    all_data = (list(self.simulation.human_data[key]) + 
                               list(self.simulation.robot_data_spline[key]) + 
                               list(self.simulation.robot_data_direct[key]))
                    if all_data:
                        y_min, y_max = min(all_data), max(all_data)
                        y_range = max(y_max - y_min, 10)  # 최소 범위 보장
                        y_center = (y_max + y_min) / 2
                        joint_axes[i].set_ylim(y_center - y_range*0.6, y_center + y_range*0.6)
            else:
                # 초기 상태: 기본 범위 설정
                joint_axes[i].set_xlim(0, 10)
                joint_axes[i].set_ylim(-100, 100)
                        
    def update_robot_visualization(self):
        """로봇 팔 시각화 업데이트"""
        # 스플라인 방법 로봇 팔
        positions_spline = self.simulation.get_robot_arm_position('spline')
        if positions_spline is not None:
            shoulder = positions_spline['shoulder']
            elbow = positions_spline['elbow']
            wrist = positions_spline['wrist']
            end = positions_spline['end']
            
            # 스플라인 방법 개별 차트
            self.lines['robot_spline_upper'].set_data([shoulder[0], elbow[0]], [shoulder[1], elbow[1]])
            self.lines['robot_spline_forearm'].set_data([elbow[0], wrist[0]], [elbow[1], wrist[1]])
            self.lines['robot_spline_hand'].set_data([wrist[0], end[0]], [wrist[1], end[1]])
            self.lines['robot_spline_joints'].set_data([shoulder[0], elbow[0], wrist[0], end[0]], 
                                                      [shoulder[1], elbow[1], wrist[1], end[1]])
            
            # 오버레이 차트 - 스플라인
            self.lines['overlay_spline_upper'].set_data([shoulder[0], elbow[0]], [shoulder[1], elbow[1]])
            self.lines['overlay_spline_forearm'].set_data([elbow[0], wrist[0]], [elbow[1], wrist[1]])
            self.lines['overlay_spline_hand'].set_data([wrist[0], end[0]], [wrist[1], end[1]])
            
            # 궤적 데이터 저장
            self.trajectory_data['spline_x'].append(end[0])
            self.trajectory_data['spline_y'].append(end[1])
        
        # 직접 매핑 방법 로봇 팔
        positions_direct = self.simulation.get_robot_arm_position('direct')
        if positions_direct is not None:
            shoulder = positions_direct['shoulder']
            elbow = positions_direct['elbow']
            wrist = positions_direct['wrist']
            end = positions_direct['end']
            
            # 직접 매핑 방법 개별 차트
            self.lines['robot_direct_upper'].set_data([shoulder[0], elbow[0]], [shoulder[1], elbow[1]])
            self.lines['robot_direct_forearm'].set_data([elbow[0], wrist[0]], [elbow[1], wrist[1]])
            self.lines['robot_direct_hand'].set_data([wrist[0], end[0]], [wrist[1], end[1]])
            self.lines['robot_direct_joints'].set_data([shoulder[0], elbow[0], wrist[0], end[0]], 
                                                      [shoulder[1], elbow[1], wrist[1], end[1]])
            
            # 오버레이 차트 - 직접 매핑
            self.lines['overlay_direct_upper'].set_data([shoulder[0], elbow[0]], [shoulder[1], elbow[1]])
            self.lines['overlay_direct_forearm'].set_data([elbow[0], wrist[0]], [elbow[1], wrist[1]])
            self.lines['overlay_direct_hand'].set_data([wrist[0], end[0]], [wrist[1], end[1]])
            
            # 궤적 데이터 저장
            self.trajectory_data['direct_x'].append(end[0])
            self.trajectory_data['direct_y'].append(end[1])
        
        # 엔드 이펙터 궤적 업데이트
        if len(self.trajectory_data['spline_x']) > 1:
            self.lines['trajectory_spline'].set_data(list(self.trajectory_data['spline_x']), 
                                                   list(self.trajectory_data['spline_y']))
            # 현재 위치 표시
            self.lines['trajectory_spline_current'].set_data([self.trajectory_data['spline_x'][-1]], 
                                                           [self.trajectory_data['spline_y'][-1]])
        
        if len(self.trajectory_data['direct_x']) > 1:
            self.lines['trajectory_direct'].set_data(list(self.trajectory_data['direct_x']), 
                                                    list(self.trajectory_data['direct_y']))
            # 현재 위치 표시
            self.lines['trajectory_direct_current'].set_data([self.trajectory_data['direct_x'][-1]], 
                                                           [self.trajectory_data['direct_y'][-1]])
        
    def update_performance_metrics(self):
        """성능 지표 업데이트"""
        metrics = self.simulation.get_performance_metrics()
        if not metrics or 'spline' not in metrics or 'direct' not in metrics:
            return
            
        rmse_spline = []
        rmse_direct = []
        jerk_spline = []
        jerk_direct = []
        
        for i, joint in enumerate(self.joint_keys):
            if joint in metrics['spline']:
                rmse_spline.append(metrics['spline'][joint]['rmse'])
                jerk_spline.append(metrics['spline'][joint]['jerk'])
                self.bars['performance_spline'][i].set_height(metrics['spline'][joint]['rmse'])
                self.bars['jerk_spline'][i].set_height(metrics['spline'][joint]['jerk'])
            else:
                rmse_spline.append(0)
                jerk_spline.append(0)
                self.bars['performance_spline'][i].set_height(0)
                self.bars['jerk_spline'][i].set_height(0)
            
            if joint in metrics['direct']:
                rmse_direct.append(metrics['direct'][joint]['rmse'])
                jerk_direct.append(metrics['direct'][joint]['jerk'])
                self.bars['performance_direct'][i].set_height(metrics['direct'][joint]['rmse'])
                self.bars['jerk_direct'][i].set_height(metrics['direct'][joint]['jerk'])
            else:
                rmse_direct.append(0)
                jerk_direct.append(0)
                self.bars['performance_direct'][i].set_height(0)
                self.bars['jerk_direct'][i].set_height(0)
        
        # 축 범위 조정
        if rmse_spline or rmse_direct:
            max_rmse = max(max(rmse_spline + rmse_direct), 1)
            self.axes['performance'].set_ylim(0, max_rmse * 1.2)
        
        if jerk_spline or jerk_direct:
            max_jerk = max(max(jerk_spline + jerk_direct), 1)
            self.axes['smoothness'].set_ylim(0, max_jerk * 1.2)
        
        # 성능 텍스트 업데이트
        avg_rmse_spline = np.mean(rmse_spline) if rmse_spline else 0
        avg_rmse_direct = np.mean(rmse_direct) if rmse_direct else 0
        avg_jerk_spline = np.mean(jerk_spline) if jerk_spline else 0
        avg_jerk_direct = np.mean(jerk_direct) if jerk_direct else 0
        
        improvement_rmse = ((avg_rmse_direct - avg_rmse_spline) / max(avg_rmse_direct, 0.001)) * 100
        improvement_jerk = ((avg_jerk_direct - avg_jerk_spline) / max(avg_jerk_direct, 0.001)) * 100
        
        self.performance_text.set_text(
            f"🎯 Performance Comparison:\n"
            f"📊 RMSE - Spline: {avg_rmse_spline:.2f}°, Direct: {avg_rmse_direct:.2f}°\n"
            f"📈 Jerk - Spline: {avg_jerk_spline:.2f}, Direct: {avg_jerk_direct:.2f}\n"
            f"⬆️ Improvement - RMSE: {improvement_rmse:.1f}%, Jerk: {improvement_jerk:.1f}%"
        )
    
    def animate(self, frame):
        """애니메이션 업데이트 함수"""
        if self.simulation.is_running:
            self.simulation.update_data()
        
        # 모든 플롯 업데이트
        self.update_joint_plots()
        self.update_robot_visualization()
        self.update_performance_metrics()
        
        # 모든 그래픽 요소 반환 (애니메이션용)
        return (list(self.lines.values()) + 
                list(self.bars['performance_spline']) + 
                list(self.bars['performance_direct']) +
                list(self.bars['jerk_spline']) + 
                list(self.bars['jerk_direct']) +
                [self.performance_text])
    
    def start_animation(self):
        """애니메이션 시작"""
        ani = animation.FuncAnimation(self.fig, self.animate, interval=50, blit=False)
        self.simulation.is_running = True
        
        # 레이아웃 최적화 - tight_layout 대신 수동 조정 사용
        # (이미 subplots_adjust에서 설정했으므로 추가 조정 불필요)
        
        # 전체 화면 모드 권장 메시지
        print("💡 Tip: Press 'f' key on the plot to toggle fullscreen mode for better viewing!")
        print("💡 Tip: Use mouse wheel to zoom in/out on individual plots")
        print("🤖 Robot Arm Comparison:")
        print("   • Top-middle: Spline method (smooth, green)")
        print("   • Top-right: Direct method (responsive, orange/purple)")  
        print("   • Middle-left: Overlay comparison")
        print("   • Middle-right: End-effector trajectory traces")
        
        plt.show()
        
        return ani

def create_visualization(simulation):
    """시각화 생성 함수"""
    visualizer = SimulationVisualizer(simulation)
    return visualizer.start_animation()