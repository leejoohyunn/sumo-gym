import os
import json
import numpy as np
from sumo_gym.envs.fmp import FMPEnv
from baselines.DQN.dqn import ReplayBuffer

# Set SUMO environment variable
os.environ['SUMO_GUI_PATH'] = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"

# Create environment
env = FMPEnv(
    mode="sumo_config",
    verbose=1,
    sumo_config_path="assets/data/cosmos/cosmos.sumocfg",
    net_xml_file_path="assets/data/cosmos/cosmos.net.xml",
    demand_xml_file_path="assets/data/cosmos/cosmos.rou.xml",
    additional_xml_file_path="assets/data/cosmos/cosmos.cs.add.xml",
    render_env=False,
)

# Initialize replay buffers like in the training script
replay_buffer_upper = {
    agent: ReplayBuffer() for agent in env.possible_agents
}
replay_buffer_lower_demand = ReplayBuffer()
replay_buffer_lower_cs = ReplayBuffer()

print("Environment agents:", env.possible_agents)
print("Replay buffer structure:")
print(f"Upper level buffers: {list(replay_buffer_upper.keys())}")
print(f"Lower demand buffer max length: {replay_buffer_lower_demand.max_len}")
print(f"Lower CS buffer max length: {replay_buffer_lower_cs.max_len}")

# Run a few steps to see what gets stored
env.reset()
step_count = 0
for agent in env.agent_iter():
    if step_count >= 10:  # Just run a few steps
        break
    
    upper_last, lower_last = env.last()
    
    # Simulate some experience data
    if len(replay_buffer_upper[agent]) == 0:  # First experience
        replay_buffer_upper[agent].push([upper_last[0], 0, None, upper_last[1]])
        replay_buffer_lower_demand.push([lower_last[0][0], 0, lower_last[0][0], 0.5])
        replay_buffer_lower_cs.push([lower_last[0][1], 0, lower_last[0][1], 0.3])
    
    # Take random action and step
    upper_action = env.action_space(agent).sample()
    lower_action = 0  # Simple action
    env.step((upper_action, lower_action))
    step_count += 1

# Examine replay buffer contents
print("\n=== REPLAY BUFFER CONTENTS ===")

for agent in list(replay_buffer_upper.keys())[:2]:  # Show first 2 agents
    print(f"\nAgent {agent} upper buffer:")
    print(f"  Buffer length: {len(replay_buffer_upper[agent])}")
    if len(replay_buffer_upper[agent]) > 0:
        print(f"  First experience: {replay_buffer_upper[agent][0]}")
        print(f"  Experience format: [state, action, next_state, reward]")

print(f"\nLower demand buffer:")
print(f"  Buffer length: {len(replay_buffer_lower_demand)}")
if len(replay_buffer_lower_demand) > 0:
    print(f"  First experience: {replay_buffer_lower_demand[0]}")

print(f"\nLower charging station buffer:")
print(f"  Buffer length: {len(replay_buffer_lower_cs)}")
if len(replay_buffer_lower_cs) > 0:
    print(f"  First experience: {replay_buffer_lower_cs[0]}")