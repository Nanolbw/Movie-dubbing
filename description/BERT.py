from transformers import BertTokenizer, BertModel
import torch
import json
import pickle
import os
import numpy as np

f = open('/home/lvyibo/new/preprocessed_data/MovieAnimation/prompt.json')
prompt_map = json.load(f)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
for parameter in model.parameters():
    parameter.requires_grad_(False)

save_path = '/home/lvyibo/new/preprocessed_data/MovieAnimation/prompt'

for json in prompt_map:
    desc = prompt_map[json][0]

    x = tokenizer(desc,
              truncation=True,
              padding='max_length',
              return_length=True,
              return_tensors='pt',
              max_length=50)

    out = model(x['input_ids'], x['attention_mask'], x['token_type_ids'])
    name = json.split('.')[0]

    np.save(os.path.join(save_path, name+'.npy'), out['last_hidden_state'])


#tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#print(out.last_hidden_state)