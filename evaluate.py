import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

#from resemblyzer import VoiceEncoder

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample, synth_multi_samples
from model import FastSpeech2Loss
from dataset import Dataset

import numpy as np

from scipy.io.wavfile import write
from tqdm import tqdm
import sys

sys.path.append("..")
# from Resemblyzer.resemblyzer import preprocess_wav
from resemblyzer import preprocess_wav

from mcd import Calculate_MCD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def acc_metric(speakers_ids, speakers_all, wav_reconstructions_utterance_embeds, \
               wav_predictions_utterance_embeds, ids2loc_map, loc2ids_map, centroids=None):
    if centroids is None:
        # Inclusive centroids (1 per speaker) (speaker_num x embed_size)
        centroids_rec = np.zeros((len(speakers_ids), wav_reconstructions_utterance_embeds.shape[1]), dtype=np.float)
        # calculate the centroids for each speaker
        counters = np.zeros((len(speakers_ids),))
        for i in range(wav_reconstructions_utterance_embeds.shape[0]):
            # calculate centroids
            centroids_rec[ids2loc_map[speakers_all[i].item()]] += wav_reconstructions_utterance_embeds[i]
            counters[ids2loc_map[speakers_all[i].item()]] += 1
        # normalize
        for i in range(len(counters)):
            centroids_rec[i] = centroids_rec[i] / counters[i]
            centroids_rec[i] = centroids_rec[i] / (np.linalg.norm(centroids_rec[i], ord=2) + 1e-5)
        # for i in range(len(wav_reconstructions_utterance_embeds)):
        #     wav_reconstructions_utterance_embeds[i] = wav_reconstructions_utterance_embeds[i] / \
        #     (np.linalg.norm(wav_reconstructions_utterance_embeds[i], ord=2) + 1e-5)
        #     wav_predictions_utterance_embeds[i] = wav_predictions_utterance_embeds[i] / \
        #     (np.linalg.norm(wav_predictions_utterance_embeds[i], ord=2) + 1e-5)
    else:
        centroids_rec = centroids

    # similarity matrix: wav_pred 512x256; centroids: num_speaker(128)x256
    # sim_matrix_pred = np.dot(wav_predictions_utterance_embeds, centroids_rec.T) \
    #     * encoder.similarity_weight.item() + encoder.similarity_bias.item()
    # sim_matrix_rec = np.dot(wav_reconstructions_utterance_embeds, centroids_rec.T) \
    #     * encoder.similarity_weight.item() + encoder.similarity_bias.item()
    sim_matrix_pred = np.dot(wav_predictions_utterance_embeds, centroids_rec.T)
    sim_matrix_rec = np.dot(wav_reconstructions_utterance_embeds, centroids_rec.T)
    # pred_locs 512x1
    pred_locs = sim_matrix_pred.argmax(axis=1)
    rec_locs = sim_matrix_rec.argmax(axis=1)
    # print(sim_matrix_rec)
    # print(rec_locs)
    # assert False

    # print(loc2ids_map)

    # calculate acc
    correct_num_pred = 0
    correct_num_rec = 0
    for i in range(len(pred_locs)):
        #print()
        f = open('./emo.txt', 'a')
        f.write(str(loc2ids_map[pred_locs[i]]) + ' - ' + str(speakers_all[i].item()) + '\n')
        if loc2ids_map[pred_locs[i]] == speakers_all[i].item():
            correct_num_pred += 1
        if loc2ids_map[rec_locs[i]] == speakers_all[i].item():
            correct_num_rec += 1
    f.close()
    eval_acc_pred = correct_num_pred / float(len(pred_locs))
    eval_acc_rec = correct_num_rec / float(len(pred_locs))

    return eval_acc_rec, eval_acc_pred


# def save_reload(ids2allloc_map_selected, wav_reconstructions_all, wav_predictions_all,\
#     utterances_per_se, samples_path, sampling_rate):
#     # save and reload val (train) samples
#     # save
#     rec_fpaths = []
#     pred_fpaths = []
#     # speakers
#     for k, v in ids2allloc_map_selected.items():
#         for i in range(utterances_per_se):
#             rec_fpath = os.path.join(samples_path, "wav_rec_{}.wav".format(v[1][i]))
#             pred_fpath = os.path.join(samples_path, "wav_pred_{}.wav".format(v[1][i]))
#             write(rec_fpath, sampling_rate, wav_reconstructions_all[v[0][i]])
#             write(pred_fpath, sampling_rate, wav_predictions_all[v[0][i]])
#             rec_fpaths.append(rec_fpath)
#             pred_fpaths.append(pred_fpath)
#     # reload
#     rec_wavs = np.array(list(map(preprocess_wav, tqdm(rec_fpaths, "Preprocessing rec wavs", len(rec_fpaths)))))
#     pred_wavs = np.array(list(map(preprocess_wav, tqdm(pred_fpaths, "Preprocessing pred wavs", len(pred_fpaths)))))

#     return rec_wavs, pred_wavs


def calculate_acc(Loss, preprocess_config2, model_config, model, fusion_model,vocoder, \
                  encoder_spk, encoder_emo, loader, logger, dataset, cal_loss=True, \
                  max_batch=None, sampling_rate=22050, samples_path=None, \
                  mcd_box_plain=None, mcd_box_dtw=None, mcd_box_adv_dtw=None):
    # Evaluation
    loss_sums = [0 for _ in range(6)]
    quick_eval = True  # evaluate only 32 samples if True, otherwise evaluate all data
    counter_batch = 0
    #
    for batchs in tqdm(loader):
        #
        wav_reconstructions_batch = []
        wav_predictions_batch = []
        tags_batch = []
        speakers_batch = []
        emotions_batch = []
        cofs_batch = []
        counter_batch += 1
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                ids, raw_texts, speaker, text, text_lens, max_text_lens, mel, mel_lens, max_mel_len, pitches, energies, durations, spk, emotion_id, emo, prompt = batch  # emos is data
                #emo = adapter(prompt, emo)
                feature = fusion_model(text, spk, emo, prompt, text_lens, max_text_lens)
                # print(feature.dtype)
                #print(feature.shape)

                batch = ids, raw_texts, speaker, feature, text_lens, max_text_lens, mel, mel_lens, max_mel_len, pitches, energies, durations, emotion_id
                output = model(*(batch[2:]), useGT=False)

                #output = model(speakers=speaker, feature=feature, src_lens=text_lens, max_src_len=max_text_lens)
                
                #output = model(*(batch[2:]))
                # if cal_loss:
                #     # Cal Loss
                #     losses = Loss(batch, output)

                #     for i in range(len(losses)):
                #         loss_sums[i] += losses[i].item() * len(batch[0])
                loss_sums = []

                if logger is not None:
                    # synthesize multiple sample for speaker and emotion accuracy calculation
                    wav_reconstructions, wav_predictions, tags, speakers, emotions = synth_multi_samples(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config2,
                    )
                    # merge
                    wav_reconstructions_batch.extend(wav_reconstructions)
                    wav_predictions_batch.extend(wav_predictions)
                    tags_batch.extend(tags)
                    speakers_batch.extend(speakers)
                    emotions_batch.extend(emotions)
        # calculate metrics
        if cal_loss:
            loss_means, acc_means_spk, acc_means_emo, avg_mcd = assess_spk_emo(
                encoder_spk=encoder_spk, encoder_emo=encoder_emo, dataset=dataset, loss_sums=loss_sums,
                cal_loss=cal_loss, sampling_rate=sampling_rate, samples_path=samples_path,
                mcd_box_plain=mcd_box_plain, mcd_box_dtw=mcd_box_dtw, mcd_box_adv_dtw=mcd_box_adv_dtw,
                wav_reconstructions_batch=wav_reconstructions_batch, wav_predictions_batch=wav_predictions_batch,
                tags_batch=tags_batch, speakers_batch=speakers_batch, emotions_batch=emotions_batch)
            if counter_batch == 1:
                acc_sums_spk = acc_means_spk
                acc_sums_emo = acc_means_emo
                sum_mcd = avg_mcd
            else:
                acc_sums_spk = list(map(lambda x: x[0] + x[1], zip(acc_sums_spk, acc_means_spk)))
                acc_sums_emo = list(map(lambda x: x[0] + x[1], zip(acc_sums_emo, acc_means_emo)))
                sum_mcd = list(map(lambda x: x[0] + x[1], zip(sum_mcd, avg_mcd)))
        else:
            acc_means_spk, acc_means_emo, avg_mcd = assess_spk_emo(
                encoder_spk=encoder_spk, encoder_emo=encoder_emo, dataset=dataset, loss_sums=loss_sums,
                cal_loss=cal_loss, sampling_rate=sampling_rate, samples_path=samples_path,
                mcd_box_plain=mcd_box_plain, mcd_box_dtw=mcd_box_dtw, mcd_box_adv_dtw=mcd_box_adv_dtw,
                wav_reconstructions_batch=wav_reconstructions_batch, wav_predictions_batch=wav_predictions_batch,
                tags_batch=tags_batch, speakers_batch=speakers_batch, emotions_batch=emotions_batch)
            if counter_batch == 1:
                acc_sums_spk = acc_means_spk
                acc_sums_emo = acc_means_emo
                sum_mcd = avg_mcd
            else:
                acc_sums_spk = list(map(lambda x: x[0] + x[1], zip(acc_sums_spk, acc_means_spk)))
                acc_sums_emo = list(map(lambda x: x[0] + x[1], zip(acc_sums_emo, acc_means_emo)))
                sum_mcd = list(map(lambda x: x[0] + x[1], zip(sum_mcd, avg_mcd)))
        # if max_batch is not None and (counter_batch==max_batch):
        # if quick_eval == True:
        if counter_batch == 2:
            break

    acc_sums_spk = list(np.array(acc_sums_spk) / counter_batch)
    acc_sums_emo = list(np.array(acc_sums_emo) / counter_batch)
    sum_mcd = list(np.array(sum_mcd) / counter_batch)
    if cal_loss:
        loss_means = []
        return batch, output, loss_means, acc_sums_spk, acc_sums_emo, sum_mcd
    else:
        return acc_sums_spk, acc_sums_emo, sum_mcd


def assess_spk_emo(encoder_spk, encoder_emo, dataset, loss_sums, cal_loss, sampling_rate, samples_path,
                   mcd_box_plain, mcd_box_dtw, mcd_box_adv_dtw,
                   wav_reconstructions_batch, wav_predictions_batch, tags_batch, speakers_batch, emotions_batch):
    # how many speaker in here (value equal to the speaker id)
    speakers_ids = torch.unique(torch.tensor(speakers_batch, dtype=torch.long))
    emotions_ids = torch.unique(torch.tensor(emotions_batch, dtype=torch.long))

    # speakers mapping
    ids2loc_map = {}
    loc2ids_map = {}
    for i in range(len(speakers_ids)):
        ids2loc_map[speakers_ids[i].item()] = i
        loc2ids_map[i] = speakers_ids[i].item()
    # emotion mapping
    ids2loc_map_emo = {}
    loc2ids_map_emo = {}
    for i in range(len(emotions_ids)):
        ids2loc_map_emo[emotions_ids[i].item()] = i
        loc2ids_map_emo[i] = emotions_ids[i].item()

    # save and reload val (train) samples
    # save
    rec_fpaths = []
    pred_fpaths = []
    for i in range(len(wav_reconstructions_batch)):
        rec_fpath = os.path.join(samples_path, "wav_rec_{}.wav".format(tags_batch[i]))
        pred_fpath = os.path.join(samples_path, "wav_pred_{}.wav".format(tags_batch[i]))
        write(rec_fpath, sampling_rate, wav_reconstructions_batch[i])
        write(pred_fpath, sampling_rate, wav_predictions_batch[i])
        rec_fpaths.append(rec_fpath)
        pred_fpaths.append(pred_fpath)
    # reload
    rec_wavs = np.array(list(map(preprocess_wav, tqdm(rec_fpaths, "Preprocessing rec wavs", len(rec_fpaths)))))
    pred_wavs = np.array(list(map(preprocess_wav, tqdm(pred_fpaths, "Preprocessing pred wavs", len(pred_fpaths)))))

    # mcd
    print("calculate MCD ...")
    for i in tqdm(range(len(rec_fpaths))):
        if i != (len(rec_fpaths) - 1):
            # mcd_box_plain.calculate_mcd(rec_fpaths[i], pred_fpaths[i], len(rec_fpaths), average=False)
            # mcd_box_dtw.calculate_mcd(rec_fpaths[i], pred_fpaths[i], len(rec_fpaths), average=False)
            # mcd_box_adv_dtw.calculate_mcd(rec_fpaths[i], pred_fpaths[i], len(rec_fpaths), average=False)
            mcd_box_plain.calculate_mcd(rec_fpaths[i], pred_fpaths[i])
            mcd_box_dtw.calculate_mcd(rec_fpaths[i], pred_fpaths[i])
            mcd_box_adv_dtw.calculate_mcd(rec_fpaths[i], pred_fpaths[i])
        else:
            # avg_mcd_plain = mcd_box_plain.calculate_mcd(
            #     rec_fpaths[i], pred_fpaths[i], len(rec_fpaths), average=True)
            # avg_mcd_dtw = mcd_box_dtw.calculate_mcd(
            #     rec_fpaths[i], pred_fpaths[i], len(rec_fpaths), average=True)
            # avg_mcd_adv_dtw = mcd_box_adv_dtw.calculate_mcd(
            #     rec_fpaths[i], pred_fpaths[i], len(rec_fpaths), average=True)
            print('get average')
            avg_mcd_plain = mcd_box_plain.calculate_mcd(
                rec_fpaths[i], pred_fpaths[i],len(rec_fpaths), average=True)
            avg_mcd_dtw = mcd_box_dtw.calculate_mcd(
                rec_fpaths[i], pred_fpaths[i],len(rec_fpaths), average=True)
            avg_mcd_adv_dtw = mcd_box_adv_dtw.calculate_mcd(
                rec_fpaths[i], pred_fpaths[i],len(rec_fpaths), average=True)
            

    # speaker and emotion: (speakers/emttion_per_batch x utterances_per_se) x embedding_dim
    # Compute the wav embedding for accuracy (spk)
    wav_reconstructions_utterance_embeds_spk = np.array(list(map(encoder_spk.embed_utterance, rec_wavs)))
    wav_predictions_utterance_embeds_spk = np.array(list(map(encoder_spk.embed_utterance, pred_wavs)))
    # Compute the wav embedding for accuracy (emo)
    wav_reconstructions_utterance_embeds_emo = np.array(list(map(encoder_emo.embed_utterance, rec_wavs)))
    wav_predictions_utterance_embeds_emo = np.array(list(map(encoder_emo.embed_utterance, pred_wavs)))

    # calcuate accuracy
    # emotion
    # centroids_emo = np.load("/mnt/cephfs/home/chenqi/workspace/Project/Resemblyzer/centroids_emo_all.npy")
    # centroids_emo = np.load("/home/qichen/Desktop/Avatar2/V2C/centroids_emo_all.npy")

    # todo；
    eval_acc_rec_emo, eval_acc_pred_emo = acc_metric(emotions_ids, emotions_batch, \
                                                     wav_reconstructions_utterance_embeds_emo,
                                                     wav_predictions_utterance_embeds_emo, \
                                                     ids2loc_map_emo, loc2ids_map_emo, centroids=None)
    # speaker
    eval_acc_rec_spk, eval_acc_pred_spk = acc_metric(speakers_ids, speakers_batch, \
                                                     wav_reconstructions_utterance_embeds_spk,
                                                     wav_predictions_utterance_embeds_spk, \
                                                     ids2loc_map, loc2ids_map)

    if cal_loss:
        #loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]
        loss_means = []
        acc_means_spk = [eval_acc_rec_spk, eval_acc_pred_spk]
        acc_means_emo = [eval_acc_rec_emo, eval_acc_pred_emo]
        avg_mcd = [avg_mcd_plain, avg_mcd_dtw, avg_mcd_adv_dtw]
        return loss_means, acc_means_spk, acc_means_emo, avg_mcd
    else:
        acc_means_spk = [eval_acc_rec_spk, eval_acc_pred_spk]
        acc_means_emo = [eval_acc_rec_emo, eval_acc_pred_emo]
        avg_mcd = [avg_mcd_plain, avg_mcd_dtw, avg_mcd_adv_dtw]
        return acc_means_spk, acc_means_emo, avg_mcd

    # if cal_loss:
    #     loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]
    #     avg_mcd = [avg_mcd_plain, avg_mcd_dtw, avg_mcd_adv_dtw]
    #     return batch, output, loss_means, avg_mcd
    # else:
    #     avg_mcd = [avg_mcd_plain, avg_mcd_dtw, avg_mcd_adv_dtw]
    #     return avg_mcd


def evaluate(model, fusion_model, step, configs, logger=None, vocoder=None, encoder_spk=None, \
             encoder_emo=None, train_samples_path=None, val_samples_path=None):
    # preprocess_config, model_config, train_config = configs
    preprocess_config, model_config, train_config, preprocess_config2 = configs

    # Get dataset
    dataset_train = Dataset(
        "train.txt", preprocess_config2, train_config, sort=False, drop_last=False
    )
    dataset_val = Dataset(
        "val.txt", preprocess_config2, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    #
    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset_train.collate_fn,
    )
    loader_train_acconly = DataLoader(
        dataset_train,
        batch_size=16,
        shuffle=False,
        pin_memory=True,
        collate_fn=dataset_train.collate_fn,
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=16,
        shuffle=False,
        pin_memory=True,
        collate_fn=dataset_val.collate_fn,
    )

    # sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    sampling_rate = preprocess_config2["preprocessing"]["audio"]["sampling_rate"]

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config2, model_config).to(device)

    # calculate the accuracy and MCD
    print("calculate training acc ...")
    # initialize MCD module
    mcd_box_plain = Calculate_MCD("plain")
    mcd_box_dtw = Calculate_MCD("dtw")
    mcd_box_adv_dtw = Calculate_MCD("adv_dtw")
    #
    acc_means_train_spk, acc_means_train_emo, avg_mcd_train = calculate_acc(Loss, preprocess_config2, model_config, \
                                                                            model, fusion_model, vocoder, encoder_spk, encoder_emo,
                                                                            loader_train_acconly, logger, dataset_train,
                                                                            cal_loss=False, \
                                                                            max_batch=2, sampling_rate=sampling_rate,
                                                                            samples_path=train_samples_path, \
                                                                            mcd_box_plain=mcd_box_plain,
                                                                            mcd_box_dtw=mcd_box_dtw,
                                                                            mcd_box_adv_dtw=mcd_box_adv_dtw)
    # avg_mcd_train = calculate_acc(Loss, preprocess_config2, model_config, \
    #     model, vocoder, encoder_spk, encoder_emo, loader_train, logger, dataset_train, cal_loss=False, \
    #     max_batch=2, sampling_rate=sampling_rate, samples_path=train_samples_path, \
    #     mcd_box_plain=mcd_box_plain, mcd_box_dtw=mcd_box_dtw, mcd_box_adv_dtw=mcd_box_adv_dtw)

    print("calculate val loss and acc ...")
    # initialize MCD module
    mcd_box_plain = Calculate_MCD("plain")
    mcd_box_dtw = Calculate_MCD("dtw")
    mcd_box_adv_dtw = Calculate_MCD("adv_dtw")
    # 
    batch, output, loss_means_val, acc_means_val_spk, acc_means_val_emo, avg_mcd_val = calculate_acc(
        Loss,preprocess_config2, model_config,model,fusion_model, vocoder,encoder_spk,encoder_emo,loader_val, logger,dataset_val,
        cal_loss=True,max_batch=2,sampling_rate=sampling_rate, samples_path=val_samples_path,mcd_box_plain=mcd_box_plain,mcd_box_dtw=mcd_box_dtw,mcd_box_adv_dtw=mcd_box_adv_dtw)
    # batch, output, loss_means_val, avg_mcd_val = calculate_acc(Loss, preprocess_config2, \
    #     model_config, model, vocoder, encoder_spk, encoder_emo, loader_val, logger, dataset_val, cal_loss=True, \
    #     max_batch=2, sampling_rate=sampling_rate, samples_path=val_samples_path, \
    #     mcd_box_plain=mcd_box_plain, mcd_box_dtw=mcd_box_dtw, mcd_box_adv_dtw=mcd_box_adv_dtw)

    # message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
    #     *([step] + [l for l in loss_means])
    # )
    # message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, \
    # Energy Loss: {:.4f}, Duration Loss: {:.4f},\
    # MCD plain|dtw|adv_dtw (train): {:.4f}|{:.4f}|{:.4f}, \
    # MCD plain|dtw|adv_dtw (val): {:.4f}|{:.4f}|{:.4f}, \
    # Id. Acc (train) (rec|pred): {:.4f}|{:.4f}, \
    # Id. Acc (val) (rec|pred): {:.4f}|{:.4f}, \
    # Emo. Acc (train) (rec|pred): {:.4f}|{:.4f}, \
    # Emo. Acc (val) (rec|pred): {:.4f}|{:.4f}".format(
    #     *([step] + [l for l in loss_means_val]), \
    #     avg_mcd_train[0], avg_mcd_train[1], avg_mcd_train[2], \
    #     avg_mcd_val[0], avg_mcd_val[1], avg_mcd_val[2], \
    #     acc_means_train_spk[0], acc_means_train_spk[1], \
    #     acc_means_val_spk[0], acc_means_val_spk[1], \
    #     acc_means_train_emo[0], acc_means_train_emo[1], \
    #     acc_means_val_emo[0], acc_means_val_emo[1]
    # )
    message = "Validation Step {}, \
    MCD plain|dtw|adv_dtw (train): {:.4f}|{:.4f}|{:.4f}, \
    MCD plain|dtw|adv_dtw (val): {:.4f}|{:.4f}|{:.4f}, \
    Id. Acc (train) (rec|pred): {:.4f}|{:.4f}, \
    Id. Acc (val) (rec|pred): {:.4f}|{:.4f}, \
    Emo. Acc (train) (rec|pred): {:.4f}|{:.4f}, \
    Emo. Acc (val) (rec|pred): {:.4f}|{:.4f}".format(
        step, avg_mcd_train[0], avg_mcd_train[1], avg_mcd_train[2], \
        avg_mcd_val[0], avg_mcd_val[1], avg_mcd_val[2], \
        acc_means_train_spk[0], acc_means_train_spk[1], \
        acc_means_val_spk[0], acc_means_val_spk[1], \
        acc_means_train_emo[0], acc_means_train_emo[1], \
        acc_means_val_emo[0], acc_means_val_emo[1]
    )


    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            # preprocess_config,
            preprocess_config2,
        )

        # log(logger, step, losses=loss_means)
        log(logger, step, accs_val_spk=acc_means_val_spk, \
            accs_train_spk=acc_means_train_spk, accs_val_emo=acc_means_val_emo, \
            accs_train_emo=acc_means_train_emo, avg_mcd_val=avg_mcd_val, \
            avg_mcd_train=avg_mcd_train)
        # log(logger, step, losses=loss_means_val, avg_mcd_val=avg_mcd_val, \
        #     avg_mcd_train=avg_mcd_train)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    return message


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
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
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # encoder_spk = VoiceEncoder().to(device)
    # encoder_emo = VoiceEncoder().to(device)
    #
    # encoder_spk.eval()
    # encoder_emo.eval()
    #
    # # Load vocoder 声码器
    # vocoder = get_vocoder(model_config, device)
    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs, vocoder, encoder_spk, encoder_emo, )

    print(message)
