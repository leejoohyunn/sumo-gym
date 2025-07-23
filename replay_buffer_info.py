import json

print("=== REPLAY BUFFER STRUCTURE IN COSMOS TRAINING ===")

print("\n1. BUFFER ORGANIZATION:")
print("   - Upper level: 8 separate buffers (one per agent)")
print("   - Agents: -12, -9, -6, -3, -11, -8, -5, -2")
print("   - Lower level: 2 shared buffers")
print("     * Demand selection buffer")  
print("     * Charging station selection buffer")

print("\n2. EXPERIENCE FORMAT:")
print("   Upper level experience: [state, action, next_state, reward]")
print("   - state: safety indicator (float)")
print("   - action: 0=demand, 1=charge, 2=idle")
print("   - next_state: updated safety indicator")
print("   - reward: performance reward")

print("\n   Lower level experience: [state_vector, action_index, new_state_vector, reward]")
print("   - Demand buffer: [demand_vector, selected_demand_id, updated_vector, reward]")
print("   - CS buffer: [cs_occupancy_vector, selected_cs_id, updated_vector, reward]")

print("\n3. BUFFER SIZES:")
print("   - Default max size: 10,000 experiences per buffer")
print("   - Total buffers: 10 (8 upper + 2 lower)")
print("   - Total capacity: ~100,000 experiences")

print("\n4. TO VIEW ACTUAL REPLAY BUFFERS:")
print("   The replay buffers are only stored in memory during training.")
print("   To save them, you would need to modify the training script.")

print("\n5. CURRENT TRAINING STATE:")
print("   From the loss/reward files, we can see:")
print("   - Agent -11 shows training instability (high loss)")
print("   - Agents -12, -9 show good convergence")
print("   - Agents -6, -3, -2 appear to have constant loss (may not be learning)")

# Create a template for what saved replay buffers would look like
template = {
    "upper_level_buffers": {
        "agent_-12": {
            "buffer_size": 1500,
            "sample_experiences": [
                [0.85, 1, 0.92, 0.15],  # [safety_state, action, next_safety, reward]
                [0.92, 0, 0.78, -0.05],
                [0.78, 2, 0.78, 0.0]
            ]
        },
        "agent_-9": {
            "buffer_size": 1200,
            "sample_experiences": [
                [0.65, 0, 0.88, 0.25],
                [0.88, 1, 0.95, 0.12]
            ]
        }
        # ... other agents
    },
    "lower_level_buffers": {
        "demand_buffer": {
            "buffer_size": 800,
            "sample_experiences": [
                [[0, 1, 0, 1, 2], 1, [0, 1, 1, 1, 2], 50.2],  # [demand_vector, chosen_demand, new_vector, reward]
                [[0, 1, 1, 1, 2], 3, [0, 1, 1, 1, 2], 25.5]
            ]
        },
        "charging_station_buffer": {
            "buffer_size": 600,
            "sample_experiences": [
                [[1, 0, 1, 3], 2, [1, 0, 0, 3], 30.8],  # [cs_occupancy, chosen_cs, new_occupancy, reward]
                [[1, 0, 0, 3], 1, [1, 1, 0, 3], 45.2]
            ]
        }
    }
}

print("\n6. EXAMPLE REPLAY BUFFER EXPORT FORMAT:")
print(json.dumps(template, indent=2))

print("\n7. TO MODIFY TRAINING TO SAVE REPLAY BUFFERS:")
print("   Add to the MADQN.train() method:")
print("   ```python")
print("   # At the end of training")
print("   replay_data = {")
print("       'upper': {agent: list(buffer.memory) for agent, buffer in self.replay_buffer_upper.items()},")
print("       'lower_demand': list(self.replay_buffer_lower_demand.memory),")
print("       'lower_cs': list(self.replay_buffer_lower_cs.memory)")
print("   }")
print("   with open('replay_buffers.json', 'w') as f:")
print("       json.dump(replay_data, f)")
print("   ```")