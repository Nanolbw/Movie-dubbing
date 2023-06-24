import torch
import torchvision.models as models

checkpoint = torch.load('E:\V2C\V2C-clean\output\ckpt\MovieAnimation\900000.pth.tar')	# 加载模型
print(checkpoint.keys())
