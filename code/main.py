import torch
print(torch.cuda.is_available())  # 应该返回 True，如果返回 False 说明 CUDA 不可用
