import os
import numpy as np
import json
import gymnasium as gym
import random

import sumo_gym
from sumo_gym.utils.fmp_utils import (
    Vertex,
    Edge,
    Demand,
    DeliveryTruck,
    TimeManager,
    generate_random_time_windows,
    get_available_deliveries,
)
from DQN.dqn import (
    QNetwork,
    ReplayBuffer,
    run_target_update,
)
from statistics import mean

suffix = "-vrptw.json"

from sumo_gym.envs.fmp import VRPTWEnv

# 🎬 녹화용 환경 생성 (GUI 켜기)
video_env = VRPTWEnv(
    mode="sumo_config",
    verbose=1,
    sumo_config_path="../assets/data/cosmos/cosmos.sumocfg",
    net_xml_file_path="../assets/data/cosmos/cosmos.net.xml",
    demand_xml_file_path="../assets/data/cosmos/cosmos.rou.xml",
    additional_xml_file_path="../assets/data/cosmos/cosmos.cs.add.xml",
    render_env=True,  # 🎥 GUI 켜기 (녹화용)
)

class VRPTWVideoDemo(object):
    def __init__(self, env):
        self.env = env
        
        # 학습된 모델 네트워크 재생성
        state_size = 105  # 현재시간(1) + 위치(1) + 적재량(1) + 배송상태(50) + 시간창정보(50×2)
        action_size = 52  # 허브(1) + 배송지(50) + 대기(1)
        
        self.q_network = QNetwork(state_size, action_size, 0.003)
        self.time_manager = TimeManager(max_episode_time=180)  # 3시간
        
        # 학습된 가중치 불러오기 (있다면)
        # self.load_trained_weights()
    
    def load_trained_weights(self):
        """학습된 가중치 불러오기 (구현 필요)"""
        # 예: self.q_network.load_weights("vrptw_model.pth")
        pass
    
    def _get_state_vector(self, agent):
        """현재 상태를 벡터로 변환"""
        agent_idx = self.env.agent_name_idx_mapping[agent]
        truck = self.env.vrptw.delivery_trucks[agent_idx]
        
        # 기본 상태 정보
        state = [
            truck.current_time / 180.0,  # 정규화된 현재 시간
            truck.location / 50.0,       # 정규화된 위치
            truck.cargo_count / 5.0,     # 정규화된 적재량
        ]
        
        # 배송 완료 상태 (50개)
        delivery_status = [1.0 if demand.is_completed else 0.0 
                          for demand in self.env.vrptw.demands]
        state.extend(delivery_status)
        
        # 시간창 정보 (50개 × 2)
        for demand in self.env.vrptw.demands:
            state.extend([
                demand.earliest_time / 180.0,  # 정규화된 시작 시간
                demand.latest_time / 180.0,    # 정규화된 종료 시간
            ])
        
        return np.array(state, dtype=np.float32)

    def _get_available_actions(self, agent):
        """현재 상태에서 가능한 액션 목록"""
        agent_idx = self.env.agent_name_idx_mapping[agent]
        truck = self.env.vrptw.delivery_trucks[agent_idx]
        available_actions = []
        
        # 허브 복귀 (항상 가능)
        available_actions.append(0)
        
        # 배송지 이동 (적재량이 있고 완료되지 않은 배송지)
        if truck.cargo_count > 0:
            for i, demand in enumerate(self.env.vrptw.demands):
                if not demand.is_completed:
                    # 시간 제약 확인
                    travel_time = self.time_manager.calculate_travel_time(
                        self.env.vrptw.vertices, self.env.vrptw.edges,
                        truck.location, demand.destination
                    )
                    arrival_time = truck.current_time + travel_time
                    
                    if arrival_time <= self.time_manager.max_episode_time:
                        available_actions.append(i + 1)  # 배송지 액션
        
        # 대기 (항상 가능)
        available_actions.append(51)
        
        return available_actions

    def _select_smart_action(self, agent):
        """스마트한 액션 선택 (학습된 모델 또는 휴리스틱)"""
        state = self._get_state_vector(agent)
        available_actions = self._get_available_actions(agent)
        
        if not available_actions:
            return 51  # 대기
        
        # 학습된 모델이 있다면 사용
        try:
            q_values = self.q_network.compute_q_values(state)
            # 가능한 액션 중에서 최고 Q값 선택
            available_q_values = [(action, q_values[action]) for action in available_actions]
            return max(available_q_values, key=lambda x: x[1])[0]
        except:
            # 모델이 없거나 오류가 있으면 휴리스틱 사용
            return self._heuristic_action(agent, available_actions)
    
    def _heuristic_action(self, agent, available_actions):
        """휴리스틱 기반 액션 선택"""
        agent_idx = self.env.agent_name_idx_mapping[agent]
        truck = self.env.vrptw.delivery_trucks[agent_idx]
        
        # 시간이 많이 지났으면 허브로 복귀
        if truck.current_time > 150:  # 2.5시간 이후
            return 0
        
        # 적재량이 있으면 가장 가까운 배송지로
        if truck.cargo_count > 0:
            delivery_actions = [a for a in available_actions if 1 <= a <= 50]
            if delivery_actions:
                # 가장 가까운 배송지 선택 (간단한 휴리스틱)
                return min(delivery_actions)
        
        # 기본적으로 허브로 복귀
        return 0
    
    def create_demo_video(self, num_episodes=3):
        """학습된 모델로 데모 영상 생성"""
        
        print("🎬" + "="*60)
        print("🚚 VRPTW Fleet Management - VIDEO DEMO 🚚")
        print("🎬" + "="*60)
        print("📹 SUMO GUI가 열립니다. 화면 녹화를 시작하세요!")
        print("🎮 GUI 조작법:")
        print("   - Space: 시뮬레이션 시작/일시정지")
        print("   - +/-: 속도 조절")
        print("   - Ctrl+A: 최대 속도")
        print("   - View → Vehicle → Show As: 차량 표시 설정")
        print("🎬" + "="*60)
        
        input("📹 화면 녹화 준비가 되면 Enter를 누르세요...")
        
        for episode in range(num_episodes):
            print(f"\n🎬 Episode {episode + 1}/{num_episodes} 시작!")
            
            self.env.reset()
            episode_step = 0
            total_deliveries = 0
            total_distance = 0
            
            print(f"🎯 Episode {episode + 1} 목표:")
            print("   - 50개 배송지 서비스")
            print("   - 시간창 제약 준수") 
            print("   - 효율적인 경로 선택")
            print("   - 5대 트럭 협력 운행")
            
            for agent in self.env.agent_iter():
                agent_idx = self.env.agent_name_idx_mapping[agent]
                truck = self.env.vrptw.delivery_trucks[agent_idx]
                
                # 스마트한 액션 선택
                action = self._select_smart_action(agent)
                
                # 액션 이름 매핑
                if action == 0:
                    action_name = "🏠 HUB"
                elif 1 <= action <= 50:
                    action_name = f"📦 DELIVERY_{action}"
                else:
                    action_name = "⏳ WAIT"
                
                print(f"   🚚 {agent} (Time: {truck.current_time:.1f}, "
                      f"Location: {truck.location}, Cargo: {truck.cargo_count}): {action_name}")
                
                # 환경에서 액션 실행
                observation, reward, done, info = self.env.last()
                self.env.step(action)
                
                episode_step += 1
                
                # 배송 완료 체크
                completed_deliveries = sum(1 for demand in self.env.vrptw.demands if demand.is_completed)
                if completed_deliveries > total_deliveries:
                    total_deliveries = completed_deliveries
                    print(f"   ✅ 배송 완료! 총 {total_deliveries}/50개")
                
                if done:
                    break
                
                # 너무 긴 에피소드 방지
                if episode_step > 500:
                    print("   ⏰ 시간 초과로 에피소드 종료")
                    break
            
            print(f"✅ Episode {episode + 1} 완료!")
            print(f"📊 Steps: {episode_step}, Deliveries: {total_deliveries}/50")
            print(f"📈 Success Rate: {total_deliveries/50*100:.1f}%")
            
            if episode < num_episodes - 1:
                input("🎬 다음 에피소드를 위해 Enter를 누르세요...")
        
        print("\n🎉" + "="*60)
        print("🎬 VRPTW 비디오 데모 완료!")
        print("📹 화면 녹화를 중지하고 파일을 저장하세요!")
        print("🎉" + "="*60)

# 실행
if __name__ == "__main__":
    print("🎬 VRPTW Fleet Management Video Demo")
    print("💡 이 스크립트는 학습 완료 후 사용하세요!")
    
    demo = VRPTWVideoDemo(video_env)
    demo.create_demo_video(num_episodes=3)
    
    video_env.close()
    print("🎊 Demo 완료!")