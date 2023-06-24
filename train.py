import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from torch.utils.data.distributed import DistributedSampler
# import torch.distributed as dist


from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss, adapter
from dataset import Dataset


from evaluate import evaluate

from torch.utils.data import ConcatDataset

import sys

sys.path.append("..")
# from Resemblyzer.resemblyzer import VoiceEncoder
from resemblyzer import VoiceEncoder
import shutil

# from scipy.io.wavfile import write


# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#os.environ['MASTER_ADDR'] = 'localhost'
#  python -m torch.distributed.launch --nproc_per_node=1 train.py --restore_step 900000 -p config/MovieAnimation/preprocess.yaml -m config/MovieAnimation/model.yaml -t config/MovieAnimation/train.yaml -p2 config/MovieAnimation/preprocess.yaml
#local_rank = [0,2]


def main(args, configs):
    print("Prepare training ...")
    

    # preprocess_config, model_config, train_config = configs
    preprocess_config, model_config, train_config, preprocess_config2 = configs

    # Get dataset
    # dataset = Dataset(
    #     "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    # )
    # 数据集
    dataset1 = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    dataset2 = Dataset("train.txt", preprocess_config2, train_config, sort=True, drop_last=True)
    datasets = [dataset1, dataset2]
    dataset = ConcatDataset(datasets)

    # 获取配置中的batchSize
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset

    assert batch_size * group_size < len(dataset)
    # 使用DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        num_workers=8,
        pin_memory=True,
        # collate_fn=dataset.collate_fn,
        collate_fn=dataset1.collate_fn,
    )

    
    # Prepare model
    #device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model, fusion_model, adapter, optimizer = get_model(args, configs, device, train=True)
    # print(model_config["Multimodal-Transformer"])
    #print(device)


    # 多GPU的计算
    # model = model.cuda()
    # fusion_model = fusion_model.cuda()
    # model = nn.DataParallel(model)
    # fusion_model = nn.DataParallel(fusion_model)

    # num_param 获取张量元素个数
    num_param = get_param_num(model) + get_param_num(fusion_model)
    print("Number of the whole model Parameters:", num_param)
    # fastSpeech2 Loss
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Load vocie encoder (compute the wav embedding for accuracy only)
    # encoder_spk = VoiceEncoder(weights_fpath=\
    #     "/home/qichen/Desktop/Avatar2/V2C/audio_encoder/MovieAnimation_bak_1567500.pt").to(device)
    # encoder_emo = VoiceEncoder(weights_fpath=\
    #     "/home/qichen/Desktop/Avatar2/V2C/audio_encoder/MovieAnimation_bak_1972500.pt").to(device)

    # resemblyzer VoiceEncoder**
    encoder_spk = VoiceEncoder().to(device)
    encoder_emo = VoiceEncoder().to(device)

    encoder_spk.eval()
    encoder_emo.eval()

    # Load vocoder 声码器
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    # train log
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    # val log
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    # tensorboard summaryWriter 使用tensorboard --logdir=xxx访问 
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # rec and gen .
    train_samples_path = os.path.join(train_log_path, "samples")
    val_samples_path = os.path.join(val_log_path, "samples")
    if os.path.exists(train_samples_path):
        # 递归删除文件
        shutil.rmtree(train_samples_path)
    if os.path.exists(val_samples_path):
        shutil.rmtree(val_samples_path)
    # 重新创建 samples文件夹
    os.makedirs(train_samples_path, exist_ok=True)
    os.makedirs(val_samples_path, exist_ok=True)

    # Training!
    step = args.restore_step + 1
    epoch = 1
    # 获取配置文件内的hyperParameter
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    # tqdm python进度条库 epoch desc 左侧的描述文字
    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        # 另一个进度条 数据集上的
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        # batches batch
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                # print(batch)
                ids, raw_texts, speaker, text, text_lens, max_text_lens, mel, mel_lens, max_mel_len, pitches, energies, durations, spk, emotion_id, emo, prompt = batch  # emos is data
                
                #emo = adapter(prompt, emo)
                
                feature = fusion_model(text, spk, emo, prompt, text_lens, max_text_lens)
                
                # print(feature.dtype)
                batch = ids, raw_texts, speaker, feature, text_lens, max_text_lens, mel, mel_lens, max_mel_len, pitches, energies, durations, emotion_id
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)

                total_loss = losses[0]

                # Backward
                total_loss = total_loss / (grad_acc_step*0.75)
                total_loss.backward()

                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                # 记录log 当训练步数到达预先设定的log_step时，调动utils文件夹下tool.py里的log function，记录loss和step
                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )
                    # 写入log
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)
                    # pyTorch.util log
                    log(train_logger, step, losses=losses)
                # 当训练步数到达预先设定的synth_step*时，调动utils文件夹下tool.py里的log function 和 synth_one_sample function
                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    # img
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]

                    # write(os.path.join(train_samples_path, "wav_rec.wav"), \
                    #     sampling_rate, wav_reconstruction)
                    # write(os.path.join(train_samples_path, "wav_pred.wav"), \
                    #     sampling_rate, wav_prediction)
                    # assert False

                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                # 当训练步数到达预先设定的val_step时，调动evaluate.py里的evaluate function来进行evaluation,并记录在log.txt
                if step % val_step == 0:
                    # 模型中有BatchNormalization和Dropout，在预测时使用model.eval()后会将其关闭以免影响预测结果。
                    model.eval()
                    # message = evaluate(model, step, configs, val_logger, vocoder)
                    message = evaluate(model, fusion_model,step, configs, val_logger, \
                                       vocoder, encoder_spk, encoder_emo, \
                                       train_samples_path, val_samples_path)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()
                    # assert False
                # 当训练步数到达预先设定的save_step时，保存训练模型
                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "fusion_model": fusion_model.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )
                # 当训练步数到达预先设定的total_step时，退出训练
                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument("-p2", "--preprocess_config2", type=str,
                        required=True, help="path to the second preprocess.yaml",
                        )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    preprocess_config2 = yaml.load(
        open(args.preprocess_config2, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    # configs = (preprocess_config, model_config, train_config)
    configs = (preprocess_config, model_config, train_config, preprocess_config2)

    main(args, configs)
