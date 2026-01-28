# Train

This project provides core model components only. Training loops are not included.

## Suggested steps
1. Prepare tokenized text and optional observations.
2. Build targets for language, policy, value, and optional world prediction.
3. Call `VAGICore.forward(..., return_loss=True)` and optimize the total loss.

