import json
from baselines.DQN.dqn import ReplayBuffer

# Create example replay buffers
replay_buffer = ReplayBuffer(max_len=1000)

# Add some example experiences
# Format: [state, action, next_state, reward]
example_experiences = [
    [0.5, 1, 0.7, 0.1],      # state=0.5, action=1, next_state=0.7, reward=0.1
    [0.7, 0, 0.3, -0.2],     # state=0.7, action=0, next_state=0.3, reward=-0.2
    [0.3, 2, 0.9, 0.5],      # state=0.3, action=2, next_state=0.9, reward=0.5
]

for exp in example_experiences:
    replay_buffer.push(exp)

print("=== REPLAY BUFFER STRUCTURE ===")
print(f"Max buffer length: {replay_buffer.max_len}")
print(f"Current buffer length: {len(replay_buffer)}")
print(f"Buffer contents: {replay_buffer}")

print("\n=== EXPERIENCE FORMAT ===")
print("Each experience is stored as: [state, action, next_state, reward]")
print(f"Example experience: {replay_buffer[0]}")

print("\n=== SAMPLING FROM BUFFER ===")
sample = replay_buffer.sample(batch_size=2)
print(f"Sample of 2 experiences: {list(sample)}")

print("\n=== MULTI-AGENT SETUP ===")
print("In the cosmos training, there are:")
print("1. Upper level replay buffers - one per agent (8 agents: -12, -9, -6, -3, -11, -8, -5, -2)")
print("2. Lower level demand replay buffer - shared across all agents")
print("3. Lower level charging station replay buffer - shared across all agents")

print("\n=== TYPICAL BUFFER SIZES ===")
print("- Upper level buffers: 10,000 experiences each")
print("- Lower level buffers: 10,000 experiences each")
print("- Total memory usage: ~80,000+ experiences across all buffers")

# Show what would be saved if we exported replay buffers
print("\n=== TO SAVE REPLAY BUFFERS ===")
print("To save replay buffer data to JSON, you would collect:")

example_agent_buffers = {
    "agent_-12": [[0.5, 1, 0.7, 0.1], [0.7, 0, 0.3, -0.2]],
    "agent_-9": [[0.3, 2, 0.9, 0.5], [0.1, 1, 0.8, 0.3]],
    # ... more agents
}

example_lower_buffers = {
    "demand": [[0.4, 0, 0.6, 0.2], [0.8, 1, 0.5, -0.1]],
    "charging_station": [[0.2, 1, 0.9, 0.4], [0.6, 0, 0.1, 0.1]]
}

print("Upper level buffers structure:")
print(json.dumps({"sample_agents": example_agent_buffers}, indent=2))
print("\nLower level buffers structure:")
print(json.dumps(example_lower_buffers, indent=2))