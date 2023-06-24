import torch
from torch import nn
from inspect import getsourcefile
import torch.nn.functional as F
import os.path as path, sys
current_dir = path.dirname(path.abspath(getsourcefile(lambda:0)))
sys.path.insert(0, current_dir[:current_dir.rfind(path.sep)])
sys.path.append('/modules')
from modules.transformer import TransformerEncoder
from transformer import Encoder
from utils.tools import get_mask_from_lengths

class MULTModel(nn.Module):
    def __init__(self, hyp_params, model_config):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params['orig_d_l'], hyp_params['orig_d_a'], hyp_params['orig_d_v']
        self.d_l, self.d_a, self.d_v = 128, 128, 128
        self.vonly = True
        self.aonly = True
        self.lonly = True
        self.num_heads = hyp_params['num_heads']
        self.layers = hyp_params['layers']
        self.attn_dropout = hyp_params['attn_dropout']
        self.attn_dropout_a = hyp_params['attn_dropout_a']
        self.attn_dropout_v = hyp_params['attn_dropout_v']
        self.relu_dropout = hyp_params['relu_dropout']
        self.res_dropout = hyp_params['res_dropout']
        self.out_dropout = hyp_params['out_dropout']
        self.embed_dropout = hyp_params['embed_dropout']
        self.attn_mask = hyp_params['attn_mask']
        self.encoder = Encoder(model_config)

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2 * self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        
        #output_dim = hyp_params.output_dim        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, 1)
        
        self.lin = nn.Linear(180, 256)


        self.linear1 = nn.Linear(50*256, 256)

        self.combine = nn.Linear(256*2,256)
        self.out = nn.Linear(256,256)
        self.out1 = nn.Linear(256,256)
        self.sigmod = nn.ReLU()

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
            
    def forward(self, x_l, x_a, x_v, prompt,src_lens, max_src_len):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #print(prompt.shape) 64,50,768
        #print(x_v.shape) 64,256
        
        prompt = self.linear1(torch.reshape(prompt, (prompt.shape[0],-1)))
        
        #prompt = self.linear2(prompt)
        #prompt = self.linear3(prompt)
        # x_v = torch.cat((x_v,prompt),dim=1)

        # x_v = self.combine(x_v)
        

        #prompt = torch.reshape(prompt, )

        # x_l = x_l.to("cuda:1")
        # x_a = x_a.to("cuda:1")
        # x_v = x_v.to("cuda:1")
        # prompt = prompt.to("cuda:1")
        # src_lens = src_lens.to("cuda:1")
        #max_src_len = max_src_len.to("cuda:1")

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        x_l = self.encoder(x_l, src_masks)
        text_encoder = x_l
        #print(text_encoder.shape) # 64,xx,256
        x_a = x_a.unsqueeze(1).expand(-1, max_src_len, -1)
        #print(x_a.shape) 64,xx,256
        # concat with vision


        x_v = x_v.unsqueeze(1).expand(-1, max_src_len, -1)
        #print(x_v.shape)
        #print(x_l.shape)
        #print(x_a.shape)
        
        
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
       
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction
            #print(last_h_l.shape)
            #last_h_l = self.lin(last_h_l)

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]
            #last_h_a = self.lin(last_h_a)

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
            #last_h_v = self.lin(last_h_v)
        
        # if self.partial_mode == 3:
        #     last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        #
        # # A residual block
        # last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        # last_hs_proj += last_hs
        #
        # output = self.out_layer(last_hs_proj)

        if self.partial_mode == 3:
            #concat = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
            concat = (last_h_l+last_h_a+last_h_v)+prompt/50
            concat = self.out(concat)
            concat = self.sigmod(concat)
            concat = self.out1(concat)
            concat = self.sigmod(concat)
            # print(torch.mean(last_h_l))
            # print(torch.mean(last_h_a))
            # print(torch.mean(last_h_v))
            # print(torch.mean(text_encoder))
            # print(torch.mean(prompt))
        #print(concat.shape)
        concat = concat.unsqueeze(1).expand(-1, max_src_len, -1)
        #print(concat.shape)
        
        return (concat + text_encoder)/2
