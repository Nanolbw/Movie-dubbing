import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim, adapter
from MultimodalTransformer.src import models

def get_model(args, configs, device, train=False):
    # (preprocess_config, model_config, train_config) = configs
    (preprocess_config, model_config, train_config, preprocess_config2) = configs

    model = FastSpeech2(preprocess_config, preprocess_config2, model_config).to(device)
    fusion_model = models.MULTModel(model_config["Multimodal-Transformer"], model_config).to(device)
    adapter_model = adapter.Adapter().to(device)
    # restore_step hyperParameter
    if args.restore_step:
        # 二进制文件 存储了weights,biases,gradients等变量
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            # .pth.tar 是torch保存模型的一种方式 以restore_step作为名称
            "{}.pth.tar".format(args.restore_step),
        )
        # 调用模型 torch.load 加载 .pth.tar
        ckpt = torch.load(ckpt_path)
        #fusion_model.load_state_dict(ckpt["fusion_model"], strict=False)
        # remove keys of pretrained model that are not in our model (i.e., embeeding layer)
        # state_dict 状态字典**
        model_dict = model.state_dict()
        # learn_speaker false whether use Embedding layer to learn speaker embedding
        if model_config["learn_speaker"]:
            speaker_emb_weight = ckpt["model"]["speaker_emb.weight"]
            s, d = speaker_emb_weight.shape
        # 将设计到speaker_emb.weight部分参数去除
        ckpt["model"] = {k: v for k, v in ckpt["model"].items() \
        if k in model_dict and k!="speaker_emb.weight"}

        # model_dict.update(ckpt["model"])
        model.load_state_dict(ckpt["model"], strict=False)
        #
        if model_config["learn_speaker"] and s <= model.state_dict()["speaker_emb.weight"].shape[0]:
            model.state_dict()["speaker_emb.weight"][:s,:] = speaker_emb_weight

    if train:
        # ScheduledOptim 为学习率封装的一个class
        scheduled_optim = ScheduledOptim(
            model, fusion_model, adapter_model, train_config, model_config, args.restore_step
        )
        # if args.restore_step:
        #     scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        fusion_model.train()
        adapter_model.train()
        return model, fusion_model, adapter_model, scheduled_optim

    model.eval()
    fusion_model.eval()
    adapter_model.eval()
    model.requires_grad_ = False
    fusion_model.requires_grad_ = False
    return model, fusion_model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    # 正如论文中所述 这里使用了Hifi-GAN
    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        # 调用配置文件 开始vocoder
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        # LJSpeech 数据集
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
