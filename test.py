import torch
print("Используемое устройство:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")
