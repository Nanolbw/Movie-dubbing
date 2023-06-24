import os
import numpy as np
from text import text_to_sequence
import yaml
from transformer import Encoder

# with open(os.path.join('./preprocessed_data/MovieAnimation', 'train.txt'), "r", encoding="utf-8") as f:
#     name = []
#     speaker = []
#     text = []
#     raw_text = []
#     for line in f.readlines():
#         n, s, t, r = line.strip("\n").split("|")
#         name.append(n)
#         speaker.append(s)
#         text.append(t)
#         raw_text.append(r)
#
# model_config = yaml.load(open('./config/MovieAnimation/model.yaml', "r"), Loader=yaml.FullLoader)
# encoder = Encoder(model_config)
#
# for idx in range(len(text)):
#     phone = np.array(text_to_sequence(text[idx], ["english_cleaners"]))



    # save_name = speaker[idx] + '-phoneme-' + name[idx]
    # save_path = os.path.join('./preprocessed_data/MovieAnimation/phoneme', save_name+'.npy')
    # np.save(save_path, )

def test(a,b=None,c=None):
    print(a,b,c)
test(1,2)