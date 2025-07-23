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
    ChargingStation,
    ElectricVehicles,
    get_dist_of_demands,
    get_dist_to_charging_stations,
    get_dist_to_finish_demands,
    get_safe_indicator,
)
from DQN.dqn import (
    QNetwork,
    LowerQNetwork_ChargingStation,
    LowerQNetwork_Demand,
    ReplayBuffer,
    run_target_update,
)
from statistics import mean

suffix = "-cosmos.json"

from sumo_gym.envs.fmp import FMPEnv

# 🎬 녹화용 환경 생성 (GUI 켜기)
video_env = FMPEnv(
    mode="sumo_config",
    verbose=1,
    sumo_config_path="assets/data/cosmos/cosmos.sumocfg",
    net_xml_file_path="assets/data/cosmos/cosmos.net.xml",
    demand_xml_file_path="assets/data/cosmos/cosmos.rou.xml",
    additional_xml_file_path="assets/data/cosmos/cosmos.cs.add.xml",
    render_env=True,  # 🎥 GUI 켜기 (녹화용)
)

class VideoMADQN(object):
    def __init__(self, env):
        self.env = env
        
        # 학습된 모델 네트워크 재생성
        self.q_principal_upper = {
            agent: QNetwork(
                1,
                env.action_space(agent).n,
                0.003,
            )
            for agent in self.env.agents
        }
        
        self.q_principal_lower_demand = LowerQNetwork_Demand(
            env.action_space_lower_demand().n_demand,
            env.action_space_lower_demand().n_demand,
            0.003,
        )
        
        self.q_principal_lower_cs = LowerQNetwork_ChargingStation(
            env.action_space_lower_cs().n_cs,
            env.action_space_lower_cs().n_cs,
            0.003,
        )
    
    def create_demo_video(self, num_episodes=3):
        """학습된 모델로 데모 영상 생성"""
        
        print("🎬" + "="*60)
        print("🚗 COSMOS Fleet Management - VIDEO DEMO 🚗")
        print("🎬" + "="*60)
        print("📹 SUMO GUI가 열립니다. 화면 녹화를 시작하세요!")
        print("🎮 GUI 조작법:")
        print("   - Space: 시뮬레이션 시작/일시정지")
        print("   - +/-: 속도 조절")
        print("   - Ctrl+A: 최대 속도")
        print("   - View → Vehicle → Show As: 차량 표시 설정")
        print("   - View → Persons → Show As: 승객 표시 설정")
        print("🎬" + "="*60)
        
        input("📹 화면 녹화 준비가 되면 Enter를 누르세요...")
        
        for episode in range(num_episodes):
            print(f"\n🎬 Episode {episode + 1}/{num_episodes} 시작!")
            
            self.env.reset()
            episode_step = 0
            total_reward = 0
            
            print(f"🎯 Episode {episode + 1} 목표:")
            print("   - 승객 픽업 서비스")
            print("   - 효율적인 충전 관리") 
            print("   - 최적 경로 선택")
            
            for agent in self.env.agent_iter():
                upper_last, lower_last = self.env.last()
                observation, reward, done, info = upper_last
                
                # 학습된 정책 사용 (또는 스마트한 휴리스틱)
                if observation == 3:
                    upper_action = 2  # 대기
                    action_name = "🛑 WAIT"
                else:
                    # 학습된 네트워크 사용
                    upper_action = self.q_principal_upper[agent].compute_argmax_q(observation)
                    action_names = ["🚖 PICKUP", "🔋 CHARGE", "🛑 WAIT"]
                    action_name = action_names[upper_action]
                
                # 하위 레벨 액션
                lower_observation, _, _, _ = lower_last
                if upper_action == 1:  # 충전
                    lower_action = self.q_principal_lower_cs.compute_argmax_q(lower_observation[1])
                    print(f"   🤖 {agent}: {action_name} at Station {lower_action}")
                elif upper_action == 0:  # 승객 픽업
                    lower_action = self.q_principal_lower_demand.compute_argmax_q(lower_observation[0])
                    print(f"   🤖 {agent}: {action_name} Passenger {lower_action}")
                else:
                    lower_action = 0
                    print(f"   🤖 {agent}: {action_name}")
                
                self.env.step((upper_action, lower_action))
                total_reward += reward
                episode_step += 1
                
                if done:
                    break
            
            print(f"✅ Episode {episode + 1} 완료!")
            print(f"📊 Steps: {episode_step}, Reward: {total_reward:.2f}")
            
            if episode < num_episodes - 1:
                input("🎬 다음 에피소드를 위해 Enter를 누르세요...")
        
        print("\n🎉" + "="*60)
        print("🎬 비디오 데모 완료!")
        print("📹 화면 녹화를 중지하고 파일을 저장하세요!")
        print("🎉" + "="*60)

# 실행
if __name__ == "__main__":
    print("🎬 COSMOS Fleet Management Video Demo")
    print("💡 이 스크립트는 학습 완료 후 사용하세요!")
    
    demo = VideoMADQN(video_env)
    demo.create_demo_video(num_episodes=3)
    
    video_env.close()
    print("🎊 Demo 완료!")