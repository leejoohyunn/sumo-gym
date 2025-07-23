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

# 시각화용 환경 생성 (GUI 켜기)
vis_env = VRPTWEnv(
    mode="sumo_config",
    verbose=1,
    sumo_config_path="../assets/data/cosmos/cosmos.sumocfg",
    net_xml_file_path="../assets/data/cosmos/cosmos.net.xml",
    demand_xml_file_path="../assets/data/cosmos/cosmos.rou.xml",
    additional_xml_file_path="../assets/data/cosmos/cosmos.cs.add.xml",
    render_env=True,  # GUI 켜기
)

class VRPTWVisualizer(object):
    def __init__(self, env):
        self.env = env
        
        # 학습된 모델 불러오기용 네트워크 생성
        state_size = 105  # 현재시간(1) + 위치(1) + 적재량(1) + 배송상태(50) + 시간창정보(50×2)
        action_size = 52  # 허브(1) + 배송지(50) + 대기(1)
        
        self.q_network = QNetwork(state_size, action_size, 0.003)
        self.time_manager = TimeManager(max_episode_time=180)  # 3시간
        
        # 학습된 가중치가 있다면 불러오기 (선택사항)
        # self.load_trained_weights()
    
    def load_trained_weights(self):
        """학습된 가중치 불러오기 (구현 필요)"""
        # 예시: 
        # try:
        #     self.q_network.load_state_dict(torch.load("vrptw_model.pth"))
        #     print("✅ 학습된 모델을 성공적으로 불러왔습니다!")
        # except:
        #     print("⚠️ 학습된 모델을 찾을 수 없습니다. 휴리스틱을 사용합니다.")
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

    def _select_action(self, agent):
        """액션 선택 (학습된 모델 또는 휴리스틱)"""
        state = self._get_state_vector(agent)
        available_actions = self._get_available_actions(agent)
        
        if not available_actions:
            return 51  # 대기
        
        # 학습된 모델 사용 시도
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
        
        # 우선순위 기반 휴리스틱
        # 1. 시간이 많이 지났으면 허브로 복귀
        if truck.current_time > 150:
            return 0
        
        # 2. 적재량이 있으면 배송 우선
        if truck.cargo_count > 0:
            delivery_actions = [a for a in available_actions if 1 <= a <= 50]
            if delivery_actions:
                # 시간창이 가장 급한 배송지 우선
                urgent_deliveries = []
                for action in delivery_actions:
                    demand_idx = action - 1
                    demand = self.env.vrptw.demands[demand_idx]
                    urgency = demand.latest_time - truck.current_time
                    urgent_deliveries.append((action, urgency))
                
                # 가장 긴급한 배송지 선택
                return min(urgent_deliveries, key=lambda x: x[1])[0]
        
        # 3. 기본적으로 허브로 복귀 (화물 적재)
        return 0
    
    def run_episode(self):
        """학습된 모델로 1 에피소드 실행"""
        print("="*70)
        print("🚚 VRPTW Fleet Management Visualization 🚚")
        print("="*70)
        print("📺 SUMO GUI Controls:")
        print("   - Space: Start/Pause simulation")
        print("   - +/-: Speed up/down")
        print("   - Ctrl+A: Maximum speed")
        print("   - View menu: Customize display options")
        print("="*70)
        
        self.env.reset()
        episode_step = 0
        completed_deliveries = 0
        truck_stats = {agent: {"distance": 0, "deliveries": 0, "time_violations": 0} 
                      for agent in self.env.possible_agents}
        
        print(f"🎯 Mission: Deliver to 50 locations using 5 trucks")
        print(f"⏰ Time limit: 3 hours (180 minutes)")
        print("-" * 70)
        
        for agent in self.env.agent_iter():
            agent_idx = self.env.agent_name_idx_mapping[agent]
            truck = self.env.vrptw.delivery_trucks[agent_idx]
            
            print(f"Step {episode_step:3d}: {agent}")
            print(f"  📍 Location: {truck.location:2d}, "
                  f"⏰ Time: {truck.current_time:5.1f}min, "
                  f"📦 Cargo: {truck.cargo_count}")
            
            # 액션 선택
            action = self._select_action(agent)
            
            # 액션 이름 및 상세 정보
            if action == 0:
                action_name = "🏠 RETURN_TO_HUB"
                action_detail = "화물 적재"
            elif 1 <= action <= 50:
                demand_idx = action - 1
                demand = self.env.vrptw.demands[demand_idx]
                action_name = f"📦 DELIVER_TO_{action:02d}"
                action_detail = f"(시간창: {demand.earliest_time:.0f}-{demand.latest_time:.0f})"
            else:
                action_name = "⏳ WAIT"
                action_detail = "대기"
            
            print(f"  🎯 Action: {action_name} {action_detail}")
            
            # 환경에서 액션 실행
            observation, reward, done, info = self.env.last()
            self.env.step(action)
            
            episode_step += 1
            
            # 통계 업데이트
            new_completed = sum(1 for demand in self.env.vrptw.demands if demand.is_completed)
            if new_completed > completed_deliveries:
                completed_deliveries = new_completed
                truck_stats[agent]["deliveries"] += 1
                print(f"  ✅ 배송 완료! 총 진행률: {completed_deliveries}/50 ({completed_deliveries/50*100:.1f}%)")
            
            # 보상 출력
            if reward != 0:
                print(f"  💰 Reward: {reward:+.1f}")
            
            if done:
                print(f"  🏁 {agent} 완료!")
                break
            
            # 진행 상황 체크
            if episode_step % 20 == 0:
                print(f"\n📊 Progress Check (Step {episode_step}):")
                print(f"   📦 Completed Deliveries: {completed_deliveries}/50")
                print(f"   ⏰ Average Time: {np.mean([truck.current_time for truck in self.env.vrptw.delivery_trucks]):.1f}min")
                print("-" * 70)
            
            # 너무 긴 에피소드 방지
            if episode_step > 1000:
                print("  ⏰ 최대 스텝 수 도달로 에피소드 종료")
                break
        
        # 최종 결과
        print("\n" + "="*70)
        print("🏁 Episode Results:")
        print(f"📊 Total Steps: {episode_step}")
        print(f"📦 Completed Deliveries: {completed_deliveries}/50")
        print(f"📈 Success Rate: {completed_deliveries/50*100:.1f}%")
        print(f"⏰ Total Time: {max(truck.current_time for truck in self.env.vrptw.delivery_trucks):.1f} minutes")
        
        # 트럭별 통계
        print("\n🚚 Truck Performance:")
        for agent, stats in truck_stats.items():
            agent_idx = self.env.agent_name_idx_mapping[agent]
            truck = self.env.vrptw.delivery_trucks[agent_idx]
            print(f"  {agent}: {stats['deliveries']} deliveries, "
                  f"Final location: {truck.location}, "
                  f"Final time: {truck.current_time:.1f}min")
        
        print("="*70)
        
        return completed_deliveries, episode_step

# 실행
if __name__ == "__main__":
    visualizer = VRPTWVisualizer(vis_env)
    
    # 여러 에피소드 실행 (원한다면)
    num_episodes = 3
    total_deliveries = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        print(f"\n🎬 Starting Episode {episode + 1}/{num_episodes}")
        deliveries, steps = visualizer.run_episode()
        total_deliveries += deliveries
        total_steps += steps
        
        if episode < num_episodes - 1:
            input("\n⏸️  Press Enter to continue to next episode...")
    
    # 전체 결과 요약
    if num_episodes > 1:
        print(f"\n🎉 Overall Performance Summary:")
        print(f"📊 Average Deliveries: {total_deliveries/num_episodes:.1f}/50")
        print(f"📈 Average Success Rate: {total_deliveries/(num_episodes*50)*100:.1f}%")
        print(f"🎯 Average Steps: {total_steps/num_episodes:.0f}")
    
    vis_env.close()
    print("\n🎉 VRPTW Visualization completed!")