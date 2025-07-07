"""
Human-Robot Trajectory Control Simulation
메인 실행 파일

사용법:
1. 필요한 라이브러리 설치: !pip install matplotlib numpy scipy
2. 파일들을 같은 폴더에 저장:
   - simulation_core.py
   - visualization.py  
   - main.py
3. main.py 실행

파일 구조:
├── simulation_core.py    # 핵심 시뮬레이션 로직
├── visualization.py      # 시각화 모듈
└── main.py              # 메인 실행 파일 (현재 파일)
"""

# 필요한 라이브러리 확인 및 임포트
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy
    print("✅ All required libraries are installed")
except ImportError as e:
    print(f"❌ Required libraries not installed: {e}")
    print("Please install with: !pip install matplotlib numpy scipy")
    exit()

# 로컬 모듈 임포트
try:
    from simulation_core import RealTimeSimulation
    from visualization import create_visualization
    print("✅ Local modules imported successfully")
except ImportError as e:
    print(f"❌ Error importing local modules: {e}")
    print("Make sure simulation_core.py and visualization.py are in the same folder")
    exit()

def print_usage_info():
    """사용법 안내"""
    print("\n" + "="*60)
    print("🤖 Human-Robot Trajectory Control Simulation")
    print("="*60)
    print("📊 VISUALIZATION GUIDE:")
    print("  • Blue line: Human motion (reference)")
    print("  • Green solid line: Spline-based robot control")
    print("  • Red dashed line: Direct mapping robot control")
    print()
    print("🎮 CONTROLS:")
    print("  • Start/Stop button: Control simulation")
    print("  • Noise Level slider: Adjust sensor noise (0.0~0.2)")
    print()
    print("📈 CHARTS EXPLANATION:")
    print("  • Top 3 charts: Individual joint comparisons")
    print("  • Bottom left: RMSE comparison (accuracy)")
    print("  • Bottom center: Robot arm visualization")
    print("  • Bottom right: Jerk comparison (smoothness)")
    print()
    print("🔍 EXPECTED RESULTS:")
    print("  • Spline method: Smoother motion, lower jerk")
    print("  • Direct method: Faster response, more noise sensitivity")
    print("="*60)

def run_simulation():
    """시뮬레이션 실행"""
    print_usage_info()
    
    try:
        # 시뮬레이션 객체 생성
        print("\n🚀 Initializing simulation...")
        simulation = RealTimeSimulation(window_size=50)
        print("✅ Simulation initialized")
        
        # 시각화 시작
        print("🎨 Starting visualization...")
        animation = create_visualization(simulation)
        print("✅ Visualization started")
        
        return simulation, animation
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        return None, None

def demo_without_gui():
    """GUI 없이 데모 실행 (디버깅용)"""
    print("🔧 Running demo without GUI...")
    
    simulation = RealTimeSimulation(window_size=20)
    
    # 10초간 시뮬레이션 실행
    for i in range(200):  # 50ms * 200 = 10초
        simulation.update_data()
        
        if i % 40 == 0:  # 2초마다 출력
            metrics = simulation.get_performance_metrics()
            if metrics and 'spline' in metrics:
                print(f"Time: {simulation.current_time:.1f}s")
                for joint in ['shoulder', 'elbow', 'wrist']:
                    if joint in metrics['spline']:
                        rmse_s = metrics['spline'][joint]['rmse']
                        rmse_d = metrics['direct'][joint]['rmse']
                        print(f"  {joint}: Spline RMSE={rmse_s:.2f}, Direct RMSE={rmse_d:.2f}")
                print()
    
    print("✅ Demo completed")

if __name__ == "__main__":
    print("Starting Human-Robot Trajectory Control Simulation...")
    
    # 환경 확인
    import sys
    if 'ipykernel' in sys.modules:
        print("📓 Running in Jupyter/Colab environment")
    else:
        print("🖥️ Running in standard Python environment")
    
    # 사용자 선택
    print("\nSelect mode:")
    print("1. Full visualization (recommended)")
    print("2. Demo without GUI (for debugging)")
    
    try:
        # GUI 환경에서는 자동으로 시각화 실행
        simulation, animation = run_simulation()
        
        if simulation is not None:
            print("\n✨ Simulation is running!")
            print("Close the plot window to stop the simulation.")
        else:
            print("\n⚠️ Falling back to demo mode...")
            demo_without_gui()
            
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Running demo mode instead...")
        demo_without_gui()

# 추가 유틸리티 함수들
def get_simulation_stats(simulation):
    """시뮬레이션 통계 출력"""
    if len(simulation.time_window) < 10:
        print("Not enough data for statistics")
        return
    
    metrics = simulation.get_performance_metrics()
    if not metrics:
        print("No metrics available")
        return
    
    print("\n📊 SIMULATION STATISTICS:")
    print("-" * 40)
    
    for joint in ['shoulder', 'elbow', 'wrist']:
        if joint in metrics['spline'] and joint in metrics['direct']:
            print(f"\n{joint.upper()} JOINT:")
            print(f"  RMSE - Spline: {metrics['spline'][joint]['rmse']:.2f}°")
            print(f"  RMSE - Direct: {metrics['direct'][joint]['rmse']:.2f}°")
            print(f"  Jerk - Spline: {metrics['spline'][joint]['jerk']:.3f}")
            print(f"  Jerk - Direct: {metrics['direct'][joint]['jerk']:.3f}")
            
            improvement = ((metrics['direct'][joint]['rmse'] - metrics['spline'][joint]['rmse']) / 
                          metrics['direct'][joint]['rmse']) * 100
            print(f"  Improvement: {improvement:.1f}%")

def save_simulation_data(simulation, filename="simulation_data.npz"):
    """시뮬레이션 데이터 저장"""
    if len(simulation.time_window) == 0:
        print("No data to save")
        return
    
    data = {
        'time': np.array(simulation.time_window),
        'human_shoulder': np.array(simulation.human_data['shoulder']),
        'human_elbow': np.array(simulation.human_data['elbow']),
        'human_wrist': np.array(simulation.human_data['wrist']),
        'robot_spline_shoulder': np.array(simulation.robot_data_spline['shoulder']),
        'robot_spline_elbow': np.array(simulation.robot_data_spline['elbow']),
        'robot_spline_wrist': np.array(simulation.robot_data_spline['wrist']),
        'robot_direct_shoulder': np.array(simulation.robot_data_direct['shoulder']),
        'robot_direct_elbow': np.array(simulation.robot_data_direct['elbow']),
        'robot_direct_wrist': np.array(simulation.robot_data_direct['wrist'])
    }
    
    np.savez(filename, **data)
    print(f"✅ Data saved to {filename}")

# 설정 상수들
SIMULATION_CONFIG = {
    'window_size': 50,
    'update_rate': 0.05,  # 50ms
    'harmonics': 3,
    'noise_level': 0.05
}

VISUALIZATION_CONFIG = {
    'figure_size': (24, 16),  # 3x3 레이아웃에 최적화
    'animation_interval': 50,  # ms
    'line_alpha': 0.8,
    'line_width': 2.5,
    'dpi': 100  # 고해상도 디스플레이용
}