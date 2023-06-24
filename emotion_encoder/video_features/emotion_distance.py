import torch
import numpy as np
import os
import json
from sklearn import preprocessing

emo_base = '../../preprocessed_data/MovieAnimation'

# angry disgust fear happy neutral sad surprise others
#num_arr = [0,0,0,0,0,0,0,0] # 756 64 305 1799 4919 572 240 1562
data_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
avg_feature = []
strength = {}

with open(os.path.join(emo_base, "emotions.json")) as f:
    emotion_map = json.load(f)


emo_feature = os.path.join(emo_base, 'emos')
for full_name in [os.path.join(emo_feature, x) for x in os.listdir(emo_feature)]:
    name = full_name.split('@')[-2].split('-')[-1] + '@' + full_name.split('@')[-1].split('.')[0]
    data = np.load(full_name)
    data_dict[emotion_map[name]].append(data)

for lists in data_dict.values():
    np_data = np.array(lists)
    avg_feature.append(np.average(np_data, axis=0))

# -1.1381254551382227e-17
# -3.469446951953614e-18
# 5.64211701039014e-18
# 4.474217303242015e-18
# -2.387916161930099e-17
# 7.763797375001095e-19
# 5.782411586589357e-19
# 1.1745477261159227e-17
# for i in data_dict.keys():
#     lists = data_dict[i]
#     np_data = np.array(lists)
#     avg = np.tile(np.array(avg_feature[2]), (len(lists),1))
#     print(np.mean(np_data-avg))
distances = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}
for i in data_dict.keys():
    lists = data_dict[i]
    center = np.array(avg_feature[4])
    for features in lists:
        distances[i].append(np.sum(features-center))


min_max_scaler = preprocessing.MinMaxScaler()
i = 0
for distance in distances.values():
    # max-min norm
    X_minMax = min_max_scaler.fit_transform(np.array(distance).reshape(-1,1))
    strength[i] = X_minMax
    i += 1

# strength_map = dict()
# list_dir = [os.path.join(emo_feature, x) for x in os.listdir(emo_feature)]
# read_list = [0,0,0,0,0,0,0,0]
# for full_name in list_dir:
#     name = full_name.split('@')[-2].split('-')[-1] + '@' + full_name.split('@')[-1].split('.')[0]
#     emo_id = emotion_map[name]
#     s = strength[emo_id][read_list[emo_id]][0]
#     read_list[emo_id] += 1
#     strength_map[name] = s

# with open(os.path.join(emo_base, "strength.json"), 'w') as f:
#    json.dump(strength_map, f)

