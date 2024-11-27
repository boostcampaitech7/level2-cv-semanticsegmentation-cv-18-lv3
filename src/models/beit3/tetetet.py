import os
import torch

print(os.environ.get('CUDA_PATH'))
print(torch.cuda.is_available())
print(torch.version.cuda)
