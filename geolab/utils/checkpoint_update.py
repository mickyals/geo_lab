import torch

# Load the checkpoint
ckpt_path = "..\..\logs\good_runs\MLP_relu_20epochs_noPE\checkpoints\last.ckpt"
checkpoint = torch.load(ckpt_path)

# Rename the keys in state_dict
state_dict = checkpoint['state_dict']
new_state_dict = {}

for key, value in state_dict.items():
    # Replace '.layer.' with '.linear.'
    new_key = key.replace('.layer.', '.linear.')
    new_state_dict[new_key] = value

# Update the checkpoint
checkpoint['state_dict'] = new_state_dict

# Save the modified checkpoint
new_ckpt_path = "..\..\logs\good_runs\MLP_relu_20epochs_noPE\checkpoints\last_fixed.ckpt"
torch.save(checkpoint, new_ckpt_path)

print(f"Fixed checkpoint saved to {new_ckpt_path}")