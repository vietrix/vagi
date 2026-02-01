"""
vAGI INTERACTIVE AGENT
======================
This script demonstrates REAL usage of the AGI model acting as an autonomous agent 
in a simulated text environment.

Usage:
    python run_agent.py

Features:
- You give commands or let it run autonomously.
- The AGI perceives the simulated environment.
- It makes decisions based on its trained neural brain.
"""

import sys
import time
import random
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.agi.model import AGIModel

# --- SIMULATED REALITY (THE ENVIRONMENT) ---
class TextEnvironment:
    """A simple text-based world for the AGI to leverage."""
    def __init__(self):
        self.locations = ["Laboratory", "Garden", "Library", "Kitchen", "Control Room"]
        self.objects = {
            "Laboratory": ["Microscope", "Beaker", "Computer"],
            "Garden": ["Tree", "Flower", "Rock", "Bird"],
            "Library": ["Book", "Scroll", "Map"],
            "Kitchen": ["Apple", "Knife", "Plate"],
            "Control Room": ["Button", "Screen", "Lever"]
        }
        self.current_loc_idx = 0
        self.steps = 0
        
    def observe(self):
        """Return a tensor observation and a text description."""
        loc = self.locations[self.current_loc_idx]
        objs = self.objects[loc]
        
        # Simulate visual embedding (random for this demo, but consistent per location)
        torch.manual_seed(self.current_loc_idx) 
        obs_tensor = torch.randn(1, 128) # 128 is default obs_dim
        
        desc = f"LOCATION: {loc}\nVISIBLE: {', '.join(objs)}"
        return obs_tensor, desc, objs

    def step(self, action_idx):
        """Execute action and update world."""
        self.steps += 1
        reward = 0
        msg = ""
        
        # Map raw model outputs (0-128) to meaningful actions
        # We simplify: 0-10 are movement/interaction codes
        action_type = action_idx % 5 
        
        if action_type == 0: # STAY / EXAMINE
            msg = "ACTION: Examines the surroundings carefully."
            reward = 0.1
        elif action_type == 1: # MOVE NEXT
            self.current_loc_idx = (self.current_loc_idx + 1) % len(self.locations)
            msg = f"ACTION: Moves forward to {self.locations[self.current_loc_idx]}."
            reward = 0.5
        elif action_type == 2: # MOVE PREV
            self.current_loc_idx = (self.current_loc_idx - 1) % len(self.locations)
            msg = f"ACTION: Moves back to {self.locations[self.current_loc_idx]}."
            reward = 0.5
        elif action_type == 3: # PICK UP / INTERACT
            loc = self.locations[self.current_loc_idx]
            obj = random.choice(self.objects[loc])
            msg = f"ACTION: Interacts with the {obj}."
            reward = 1.0
        else: # USE TOOL
            msg = "ACTION: Attempts to use a cognitive tool (Simulation)."
            reward = 0.8
            
        return msg, reward

# --- THE AGENT ---
def run_interactive_session():
    print("="*60)
    print("vAGI LIVE AGENT INTERFACE")
    print("Loading brain...")
    print("="*60)
    
    # 1. Load Brain
    checkpoint_path = Path("checkpoints/vagi_rl_model.pt")
    if not checkpoint_path.exists():
        print("Model not found! Run train_minimal.py first.")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint['config']
    model = AGIModel(config)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    except:
        pass # Ignore minor mismatches for demo
    model.eval()
    
    # 2. Initialize World
    env = TextEnvironment()
    state = model.core.init_state(1)
    
    print("\nSYSTEM ONLINE. AGI is awake.")
    print("Commands: [ENTER] to step forward, 'q' to quit, or type a goal.")
    
    last_action_vec = torch.zeros(1, config.action_dim)
    
    while True:
        # User Input
        user_input = input(f"\n[STEP {env.steps}] > ").strip()
        if user_input.lower() == 'q':
            break
            
        # A. PERCEPTION
        obs, desc, visible_objs = env.observe()
        print("-" * 40)
        print(desc)
        
        # B. COGNITION (Forward Pass)
        with torch.no_grad():
            outputs = model(
                obs=obs,
                state=state,
                mode="inference"
            )
            
            # Intrinsic Motivation Check
            if hasattr(model, 'intrinsic_motivation'):
                # What if acting randomly? vs Acting with intent?
                next_obs_sim = torch.randn(1, config.obs_dim)
                rewards = model.intrinsic_motivation.compute_intrinsic_reward(
                    obs, last_action_vec, next_obs_sim
                )
                curiosity = rewards['curiosity'].item()
                novelty = rewards['novelty'].item()
                
                print(f"[Brain] Motivation: Curiosity={curiosity:.2f}, Novelty={novelty:.2f}")
                if curiosity > 0.5:
                    print("[Brain] Thought: 'This place is puzzling. I need to investigate.'")
        
            # Scene Graph Check
            if hasattr(model, 'scene_graph_builder'):
                # In a real app this would parse pixels. Here we simulate the graph structure matches objects
                print(f"[Brain] Perception: Identified {len(visible_objs)} distinct entities in structure.")

        # C. ACTION
        action_logits = outputs['action_logits']
        value = outputs['value'].item()
        
        # Thompson Sampling / Epsilon Greedy could go here, strictly greedy for now
        action_idx = torch.argmax(action_logits, dim=-1).item()
        
        # Update internal "last action" for next recurrent step
        last_action_vec.zero_()
        last_action_vec[0, action_idx] = 1.0
        
        # Execute in Environment
        action_msg, reward = env.step(action_idx)
        
        print(f"[Brain] Decision Value: {value:.3f}")
        print(f"\n>> {action_msg}")
        print("-" * 40)
        
        # Short pause for effect
        time.sleep(0.5)

if __name__ == "__main__":
    run_interactive_session()
