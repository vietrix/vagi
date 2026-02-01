"""Debug test to find exact error location."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch

print("Starting debug test...")

try:
    print("\n[1] Import config...")
    from core.agi.config import load_agi_small_config
    config = load_agi_small_config()
    config.use_grounded_language = False
    config.use_metacognition = False
    print("  OK")
    
    print("\n[2] Import AGIModel...")
    from core.agi.model import AGIModel
    print("  OK")
    
    print("\n[3] Create model...")
    model = AGIModel(config)
    print("  OK - Model created successfully!")
    
    print(f"\n  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
