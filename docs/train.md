# Train

This project provides minimal training and evaluation scripts for the core model.

## Suggested steps
1. Prepare tokenized text and optional observations.
2. Build targets for language, policy, value, and optional world prediction.
3. Call `VAGICore.forward(..., return_loss=True)` and optimize the total loss.

## Scripts

### Train on random data
```bash
python -m scripts.train --epochs 1 --steps-per-epoch 2 --batch-size 2 --seq-len 4
```

### Train on a saved tensor dataset
```bash
python -m scripts.make_dummy_data --output data.pt
python -m scripts.train --data data.pt --with-obs --with-world
```

### Checkpointing (safetensors)
```bash
python -m scripts.train --save checkpoints/run1
python -m scripts.train --resume checkpoints/run1 --save checkpoints/run2
```

### Evaluate
```bash
python -m scripts.eval --data data.pt --with-obs --with-world --checkpoint checkpoints/run1
```
