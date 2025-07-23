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

# VRPTW 환경 설정
env = VRPTWEnv(
    mode="sumo_config",
    verbose=1,
    sumo_config_path="../assets/data/cosmos/cosmos.sumocfg",
    net_xml_file_path="../assets/data/cosmos/cosmos.net.xml",
    demand_xml_file_path="../assets/data/cosmos/cosmos.rou.xml",
    additional_xml_file_path="../assets/data/cosmos/cosmos.cs.add.xml",
    render_env=False,
)


class VRPTW_DQN(object):
    def __init__(
        self,
        env,
        lr=0.003,
        batch_size=8,
        tau=50,
        episodes=200,
        gamma=0.95,
        epsilon=1.0,
        decay_period=25,
        decay_rate=0.95,
        min_epsilon=0.01,
        initial_step=100,
    ):
        self.env = env
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_period = decay_period
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.initial_step = initial_step

        # 공유 네트워크 구조 (모든 트럭이 같은 네트워크 사용)
        # 실제 demands 수는 18개이므로: 3 + 18 + 18×2 = 57차원이 되어야 함
        # 하지만 87차원이 나오고 있으므로 실제로는 더 많은 demands가 있을 수 있음
        state_size = 153  # 고정된 상태 벡터 크기 (3 + 50 + 100)
        action_size = 52  # 허브(1) + 배송지(50) + 대기(1)
        
        self.q_principal = QNetwork(state_size, action_size, self.lr)
        self.q_target = QNetwork(state_size, action_size, self.lr)
        
        # 각 에이전트별 리플레이 버퍼 (경험은 개별 저장)
        self.replay_buffers = {
            agent: ReplayBuffer() for agent in self.env.possible_agents
        }
        
        self.total_steps = {agent: 0 for agent in self.env.possible_agents}
        self.time_manager = TimeManager(max_episode_time=180)  # 3시간

    def _initialize_output_files(self):
        """결과 저장용 파일 초기화"""
        files = ["reward", "loss", "metrics", "delivery_success"]
        
        for file_name in files:
            full_name = file_name + suffix
            if os.path.exists(full_name):
                os.remove(full_name)
            with open(full_name, "w") as f:
                f.write("{")

    def _wrap_up_output_files(self):
        """결과 파일 마무리"""
        files = ["reward", "loss", "metrics", "delivery_success"]
        
        for file_name in files:
            full_name = file_name + suffix
            with open(full_name, "a") as f:
                f.write("}")

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
        
        # 배송 완료 상태 (50개로 패딩)
        max_demands = 50
        delivery_status = []
        for i in range(max_demands):
            if i < len(self.env.vrptw.demands):
                delivery_status.append(1.0 if self.env.vrptw.demands[i].is_completed else 0.0)
            else:
                delivery_status.append(0.0)  # 패딩
        state.extend(delivery_status)
        
        # 시간창 정보 (50개 × 2 = 100차원으로 패딩)
        for i in range(max_demands):
            if i < len(self.env.vrptw.demands):
                demand = self.env.vrptw.demands[i]
                state.extend([
                    demand.earliest_time / 180.0,  # 정규화된 시작 시간
                    demand.latest_time / 180.0,    # 정규화된 종료 시간
                ])
            else:
                state.extend([0.0, 0.0])  # 패딩
        
        return state  # 리스트로 반환하여 해시 문제 방지

    def _calculate_reward(self, agent, action, delivery_completed=False, arrival_time=None):
        """VRPTW 보상 계산"""
        reward = 0
        
        if action == 0:  # 허브 복귀
            reward += 20
            
        elif 1 <= action <= 50:  # 배송지 이동
            demand_idx = action - 1
            demand = self.env.vrptw.demands[demand_idx]
            
            if delivery_completed and arrival_time is not None:
                # 배송 완료 보상
                reward += self.time_manager.calculate_delivery_reward(arrival_time, demand)
                
                # 시간창 위반 페널티
                if arrival_time > demand.latest_time:
                    penalty = self.time_manager.calculate_time_violation_penalty(arrival_time, demand)
                    reward -= penalty
            else:
                # 이동만 한 경우 이동 비용
                reward -= 1
                
        elif action == 51:  # 대기
            reward -= 0.5
            
        return reward

    def _generate_action(self, agent, state):
        """액션 생성 (epsilon-greedy)"""
        if np.random.rand() < self.epsilon:
            # 탐험: 랜덤 액션
            available_actions = self._get_available_actions(agent)
            return random.choice(available_actions) if available_actions else 51  # 대기
        else:
            # 활용: Q-네트워크 기반 액션
            q_values = self.q_principal.compute_q_values(state)
            available_actions = self._get_available_actions(agent)
            
            # 가능한 액션 중에서 최고 Q값 선택
            if available_actions:
                available_q_values = [(action, q_values[action]) for action in available_actions]
                return max(available_q_values, key=lambda x: x[1])[0]
            else:
                return 51  # 대기

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

    def _update_network(self, agent, state, action, next_state, reward, done):
        """Q-네트워크 업데이트"""
        # 리플레이 버퍼에 경험 저장
        self.replay_buffers[agent].push([state, action, next_state, reward])
        
        # 네트워크 업데이트 조건 확인
        if (self.total_steps[agent] % 10 == 0 and 
            self.total_steps[agent] > self.initial_step and
            len(self.replay_buffers[agent]) >= self.batch_size):
            
            # 배치 샘플링
            samples = self.replay_buffers[agent].sample(self.batch_size)
            states, actions, next_states, rewards = zip(*samples)
            
            # Q값 계산
            states = np.array(states)
            next_states = np.array(next_states)
            
            targets = rewards + self.gamma * self.q_target.compute_max_q(next_states)
            loss = self.q_principal.train(states, actions, targets)
            
            # 타겟 네트워크 업데이트
            if self.total_steps[agent] % self.tau == 0:
                run_target_update(self.q_principal, self.q_target)
                
            return loss
        
        return 0

    def train(self):
        """학습 실행"""
        self._initialize_output_files()
        
        for episode in range(self.episodes):
            self.env.reset()
            episode_rewards = {agent: 0 for agent in self.env.possible_agents}
            episode_losses = {agent: [] for agent in self.env.possible_agents}
            episode_steps = 0
            
            # Epsilon 감소
            if episode % self.decay_period == 0:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
            
            # 트럭별 순차 출발 시간 설정
            start_times = {f"truck_{i}": i * 2 for i in range(5)}  # 2분 간격
            
            for agent in self.env.agent_iter():
                agent_idx = self.env.agent_name_idx_mapping[agent]
                truck = self.env.vrptw.delivery_trucks[agent_idx]
                
                # 출발 시간 확인
                if truck.current_time < start_times[agent]:
                    truck.current_time = start_times[agent]
                
                # 현재 상태 획득
                state = self._get_state_vector(agent)
                
                # 액션 선택
                action = self._generate_action(agent, state)
                
                # 환경에서 액션 실행
                observation, reward, done, info = self.env.last()
                calculated_reward = self._calculate_reward(agent, action)
                
                # 다음 상태
                self.env.step(action)
                next_state = self._get_state_vector(agent)
                
                # 상태를 일관된 형태로 변환 (numpy 배열 방지)
                if hasattr(state, 'tolist'):
                    state = state.tolist()
                if hasattr(next_state, 'tolist'):
                    next_state = next_state.tolist()
                
                # 네트워크 업데이트
                loss = self._update_network(agent, state, action, next_state, calculated_reward, done)
                if loss > 0:
                    episode_losses[agent].append(loss)
                
                # 통계 업데이트
                episode_rewards[agent] += calculated_reward
                self.total_steps[agent] += 1
                episode_steps += 1
                
                # 에피소드 종료 조건 확인
                all_delivered = all(demand.is_completed for demand in self.env.vrptw.demands)
                time_exceeded = any(truck.current_time > 180 for truck in self.env.vrptw.delivery_trucks)
                
                if all_delivered or time_exceeded:
                    break
            
            # 에피소드 결과 저장
            self._save_episode_results(episode, episode_rewards, episode_losses, episode_steps)
            
            if episode % 10 == 0:
                print(f"Episode {episode}: Average reward = {np.mean(list(episode_rewards.values())):.2f}, "
                      f"Epsilon = {self.epsilon:.3f}")
        
        self._wrap_up_output_files()

    def _save_episode_results(self, episode, rewards, losses, steps):
        """에피소드 결과를 파일에 저장"""
        # 보상 저장
        reward_data = {episode: {agent: reward for agent, reward in rewards.items()}}
        with open("reward" + suffix, "a") as f:
            if episode > 0:
                f.write(",")
            f.write(json.dumps(reward_data)[1:-1])
        
        # 손실 저장
        loss_data = {episode: {agent: np.mean(loss) if loss else 0 for agent, loss in losses.items()}}
        with open("loss" + suffix, "a") as f:
            if episode > 0:
                f.write(",")
            f.write(json.dumps(loss_data)[1:-1])
        
        # 메트릭 저장
        delivered_count = sum(1 for demand in self.env.vrptw.demands if demand.is_completed)
        metric_data = {
            episode: {
                "delivered_count": delivered_count,
                "total_demands": len(self.env.vrptw.demands),
                "success_rate": delivered_count / len(self.env.vrptw.demands),
                "episode_steps": steps,
            }
        }
        with open("metrics" + suffix, "a") as f:
            if episode > 0:
                f.write(",")
            f.write(json.dumps(metric_data)[1:-1])


# 학습 실행
if __name__ == "__main__":
    dqn = VRPTW_DQN(env=env)
    dqn.train()