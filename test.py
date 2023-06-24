import torch
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'


print(torch._C._cuda_getDeviceCount())

print(torch.__version__)

print(torch.cuda.is_available())
# def process_meta(filename):
#     with open(
#          filename, "r", encoding="utf-8"
#     ) as f:
#         name = []
#         speaker = []
#         text = []
#         raw_text = []
#         for line in f.readlines():
#             n, s, t, r = line.strip("\n").split("|")
#             name.append(n)
#             speaker.append(s)
#             text.append(t)
#             raw_text.append(r)
#         return name, speaker, text, raw_text
    
# basename, speaker, _, _ = process_meta('./preprocessed_data/MovieAnimation/train.txt')

# for idx in range(len(basename)):
#     pro_path = os.path.join(
#              './preprocessed_data/MovieAnimation/emos',
#             "{}-emo-{}.npy".format(speaker[idx], basename[idx])
#         )
#     # if not os.path.exists(pro_path):
#     #     print(name)
#     emo = np.load(pro_path)
#     print(np.mean(emo))
#     prompt = np.load(pro_path)/home/lvyibo/new/preprocessed_data/MovieAnimation/emos/Bossbaby@BossBaby-emo-Bossbaby@BossBaby_00_0191_00.npy
# emos = np.load('./preprocessed_data/MovieAnimation/emos/Bossbaby@BossBaby-emo-Bossbaby@BossBaby_00_0191_00.npy')
# print(emos)