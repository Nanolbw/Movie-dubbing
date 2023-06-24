
# V2C - PyTorch Implementation

This is a PyTorch implementation of [**V2C: Visual Voice Cloning**](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_V2C_Visual_Voice_Cloning_CVPR_2022_paper.pdf). 
This project is based on [FastSpeech2](https://github.com/ming024/FastSpeech2). Feel free to use/modify the code.


# Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

# Dataset

The link of raw data can be found from [here](https://pan.baidu.com/s/1vpwf0ZqNWxZrtf6m_zIUNg), password: vvhn

# Data Preparation

There are two ways to obtain the features of V2C dataset: 1) Directly download the features from [here](https://drive.google.com/drive/folders/1E8ToFYfiofZauRCITNu1VPpdPWzv8SC-?usp=sharing); 2) Process features by ourselves.

## 1) Download Feature Directly

Please download all the features (.zip) and json files from [here](https://drive.google.com/drive/folders/1E8ToFYfiofZauRCITNu1VPpdPWzv8SC-?usp=sharing) and unzip them in the folder "./preprocessed_data/MovieAnimation"

## 2) Process Features by Ourselves

### Preprocessing

First, run 
```
python3 prepare_align.py config/MovieAnimation/preprocess.yaml
```
for some preparations.

As described in the paper, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences.
Alignments of the supported datasets are provided [here](https://drive.google.com/file/d/1IkIOZUwlDHNWH4bLW5R96_EO69pXm5hC/view?usp=sharing).
You have to unzip the files in "preprocessed_data/MovieAnimation/TextGrid/".

After that, run the preprocessing script by
```
python3 preprocess.py config/MovieAnimation/preprocess.yaml
```

Alternately, you can align the corpus by yourself.
Download the official MFA package and run
```
./montreal-forced-aligner/bin/mfa_align raw_data/MovieAnimation/ lexicon/librispeech-lexicon.txt english preprocessed_data/MovieAnimation
```
or
```
./montreal-forced-aligner/bin/mfa_train_and_align raw_data/MovieAnimation/ lexicon/librispeech-lexicon.txt preprocessed_data/MovieAnimation
```

to align the corpus and then run the preprocessing script.
```
python3 preprocess.py config/MovieAnimation/preprocess.yaml
```

### Speaker Encoder

python ./speaker_encoder/speaker_encoder.py

### Emotion Encoder

python ./emotion_encoder/video_features/emotion_encoder.py

# Training and evaluating

Download the checkpoints 900000.pth.tar from [here](https://drive.google.com/drive/folders/1E8ToFYfiofZauRCITNu1VPpdPWzv8SC-?usp=sharing) and put them in "./output/ckpt/MovieAnimation/"

Train your model with
```
python3 train.py --restore_step 900000 -p config/MovieAnimation/preprocess.yaml -m config/MovieAnimation/model.yaml -t config/MovieAnimation/train.yaml -p2 config/MovieAnimation/preprocess.yaml
```

Quickly evaluation: set "quick_eval = True" in evaluate.py for only evaluating 32 samples

Full evaluation: set "quick_eval = False" in evaluate.py for evaluating all samples

#Tensorboard

Use
```
tensorboard --logdir output/log/MovieAnimation
```

to serve TensorBoard on your localhost.
The loss curves, mcd curves, synthesized mel-spectrograms, and audios are shown.



# References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [FastSpeech 2 implementation](https://github.com/ming024/FastSpeech2)

# Citation
```
@inproceedings{chen2022v2c,
  title={V2C: Visual Voice Cloning},
  author={Chen, Qi and Tan, Mingkui and Qi, Yuankai and Zhou, Jiaqiu and Li, Yuanqing and Wu, Qi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21242--21251},
  year={2022}
}

```



