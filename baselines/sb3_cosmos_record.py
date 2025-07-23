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

# 시각화용 환경 생성 (GUI 켜기)
vis_env = FMPEnv(
    mode="sumo_config",
    verbose=1,
    sumo_config_path="assets/data/cosmos/cosmos.sumocfg",
    net_xml_file_path="assets/data/cosmos/cosmos.net.xml",
    demand_xml_file_path="assets/data/cosmos/cosmos.rou.xml",
    additional_xml_file_path="assets/data/cosmos/cosmos.cs.add.xml",
    render_env=True,  # GUI 켜기
)

class MADQNVisualizer(object):
    def __init__(self, env):
        self.env = env
        
        # 학습된 모델 불러오기용 네트워크 생성
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
        
        # 학습된 가중치가 있다면 불러오기 (선택사항)
        # self.load_trained_weights()
    
    def load_trained_weights(self):
        """학습된 가중치 불러오기 (구현 필요)"""
        pass
    
    def run_episode(self):
        """학습된 모델로 1 에피소드 실행"""
        print("="*60)
        print("🚗 COSMOS Fleet Management Visualization 🚗")
        print("="*60)
        print("📺 SUMO GUI Controls:")
        print("   - Space: Start/Pause simulation")
        print("   - +/-: Speed up/down")
        print("   - Ctrl+A: Maximum speed")
        print("="*60)
        
        self.env.reset()
        episode_step = 0
        total_reward = 0
        
        for agent in self.env.agent_iter():
            upper_last, lower_last = self.env.last()
            observation, reward, done, info = upper_last
            
            print(f"Step {episode_step}: Agent {agent}")
            print(f"  Observation: {observation}, Reward: {reward}")
            
            # 상위 레벨 액션 결정
            if observation == 3:
                upper_action = 2  # 대기
                action_name = "WAIT"
            else:
                # 학습된 네트워크 사용 (또는 랜덤)
                upper_action = self.q_principal_upper[agent].compute_argmax_q(observation)
                action_name = ["PICKUP", "CHARGE", "WAIT"][upper_action]
            
            # 하위 레벨 액션 결정
            lower_observation, _, _, _ = lower_last
            if upper_action == 1:  # 충전
                lower_action = self.q_principal_lower_cs.compute_argmax_q(lower_observation[1])
                print(f"  Action: {action_name} at charging station {lower_action}")
            elif upper_action == 0:  # 승객 픽업
                lower_action = self.q_principal_lower_demand.compute_argmax_q(lower_observation[0])
                print(f"  Action: {action_name} passenger {lower_action}")
            else:
                lower_action = 0
                print(f"  Action: {action_name}")
            
            self.env.step((upper_action, lower_action))
            total_reward += reward
            episode_step += 1
            
            if done:
                break
        
        print("="*60)
        print(f"🏁 Episode completed!")
        print(f"📊 Total steps: {episode_step}")
        print(f"🎯 Total reward: {total_reward}")
        print("="*60)
        
        return total_reward

# 실행
if __name__ == "__main__":
    visualizer = MADQNVisualizer(vis_env)
    
    # 여러 에피소드 실행 (원한다면)
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\n🎬 Starting Episode {episode + 1}/{num_episodes}")
        reward = visualizer.run_episode()
        
        if episode < num_episodes - 1:
            input("Press Enter to continue to next episode...")
    
    vis_env.close()
    print("\n🎉 Visualization completed!")