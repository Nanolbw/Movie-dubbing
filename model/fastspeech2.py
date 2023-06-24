import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    # def __init__(self, preprocess_config, model_config):
    def __init__(self, preprocess_config, preprocess_config2, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        # encoder variance_adaptor decoder
        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        # 全连接
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        # postnet在Tacotron中有讲到
        self.postnet = PostNet()

        # self.speaker_emb = None
        self.n_speaker = 1
        # multi_sperker = True
        if model_config["multi_speaker"]:
            # 读取./preprocessed_data/MovieAnimation下json文件
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                # n_speaker = len(json.load(f))
                # speaker数量*
                self.n_speaker = len(json.load(f))
            with open(
                os.path.join(
                    preprocess_config2["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                self.n_speaker += len(json.load(f))
            # speaker embedding layer
            self.speaker_emb = nn.Embedding(
                self.n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

        self.n_emotion = 1
        # with_emotion True
        if model_config["with_emotion"]:
            self.n_emotion = preprocess_config["preprocessing"]["emotion"]["n_emotion"]
            self.emotion_emb = nn.Embedding(
                self.n_emotion+1,
                model_config["transformer"]["encoder_hidden"],
                padding_idx=self.n_emotion,
            )

    def forward(
        self,
        speakers,
        feature,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        emotions=None,
        #
        useGT=True,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        
    ):
        
        src_masks = get_mask_from_lengths(src_lens, max_src_len) # tensor of True and False 16x249
        # mel_masks: 16x1941
        if useGT:
            mel_masks = (
                get_mask_from_lengths(mel_lens, max_mel_len)
                if mel_lens is not None
                else None
            )
        else:
            mel_masks = None
            max_mel_len = None
            mel_lens = None



        # output = self.encoder(texts, src_masks)
        output = feature

        # print(speakers) # 16x1
        # print(output.size()) # 16x165x256
        # print(self.speaker_emb(speakers).size()) # 16x256
        # print(self.speaker_emb(speakers).unsqueeze(1).expand(
        #         -1, max_src_len, -1).size()) # 16x165x256
        # if self.speaker_emb is not None:
        # if self.n_speaker > 1:
        #     if self.model_config["learn_speaker"]:
        #         output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
        #             -1, max_src_len, -1)
        #     else:
        #         output = output + spks.unsqueeze(1).expand(-1, max_src_len, -1)
        # if self.n_emotion > 1:
        #     if self.model_config["learn_emotion"]:
        #         output = output + self.emotion_emb(emotions).unsqueeze(1).expand(
        #             -1, max_src_len, -1)
        #     else:
        #         output = output + emos.unsqueeze(1).expand(-1, max_src_len, -1)
        
        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
            useGT
        )
        
        # print(output.shape) # 16x1877x256 20x538x256 20x54 20x54 20x54 20x54
        # print(p_predictions.shape) # 16x236
        # print(e_predictions.shape) # 16x236
        # print(log_d_predictions.shape) # 16x236
        # print(d_rounded.shape) # 16x236
        # print(mel_masks.shape) # True and False; 16x1877
        # assert False
        #print('11')

        output, mel_masks = self.decoder(output, mel_masks)
       
        output = self.mel_linear(output)
        

        postnet_output = self.postnet(output) + output

        
        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )