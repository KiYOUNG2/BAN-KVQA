# 한국어 시각 질의응답을 위한 Bilinear Attention Networks (BAN) 데모
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch 1.9.0](https://img.shields.io/badge/pytorch-1.9.0-green.svg?style=plastic)
![cuDNN 8.0](https://img.shields.io/badge/cudnn-8.0-green.svg?style=plastic)

본 레포는 [SKTBrain/BAN-KVQA](https://github.com/SKTBrain/BAN-KVQA)의 비공식 Inference Pipeline 데모를 제공합니다.  
네이버 커넥트재단의 교육과정 [Boostcamp AI-Tech](https://boostcamp.connect.or.kr/program_ai.html) 2기 KiYOUNG2 팀의 최종 프로젝트를 위한 VQA 모듈의 베이스라인으로 활용하기위해 작성되었습니다.  
원본 코드는 @jnhwkim의 [SKTBrain/BAN-KVQA](https://github.com/SKTBrain/BAN-KVQA)임을 밝힙니다. 코드 사용을 허락해주신 김진화님께 감사드립니다.

### Demo
![Demo](docs/assets/img/demo.gif)  

### 변경 사항
- huggingface transformers `v4.12.x` 지원
- 현재 다운로드 불가능한 `glove-rg`, `word2vec-pkb` 제거
- arguments 입력 방식 변경 : `argparse` 모듈 대신 `configs/` 경로 내 `*.yaml` 파일 사용
- Object Detection Module 포함 : `bottom-up-attention`을 submodule로 추가

### Model Overview
![Overview of bilinear attention networks](docs/assets/img/ban_overview.png)  

### 미리 준비할 사항

#### Requirements
```
pytorch >= 1.9.0
torchvision >= 0.10.0
transformers >= 4.12.0
jpype1 >= 1.3.0
konlpy >= 0.5.2
opencv-python >= 4.5.4.60
scikit-image >= 0.19.0
```

#### submodule 세팅(bottom-up-attention)
inference demo를 위해 bottom-up-attention을 사용합니다. 기존 caffe model을 pytorch로 변환한 [MILVLG/bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch)를 submodule로 사용합니다. 

```
# Add submodule : bottom-up-attention.pytorch
git submodule add https://github.com/KiYOUNG2/bottom-up-attention.pytorch bottom_up_attention_pytorch
git submodule update --init --recursive

# Setup bottom-up-attention
cd bottom_up_attention_pytorch
cd detectron2
pip install -e .
cd ..
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install
cd ..
python setup.py build develop

# Download bottom-up-attention model pretrained weights
wget https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EaXvCC3WjtlLvvEfLr3oa8UBLA21tcLh4L8YLbYXl6jgjg?download=1 -O bua-caffe-frcn-r101_with_attributes.pth
```

### 데모를 위한 데이터 세팅

- VQA 모델의 pretrained weight와 모델 세팅을 위한 바이너리 파일은 [여기](https://drive.google.com/drive/u/0/folders/1qq_mh3HBCe0WALvIjbD4XnTmgjgjfVNH)에서 다운로드 받으실 수 있습니다.

<br />

1. pretrained weight 파일(`ban-kvqa-roberta-base-rnn.pth`, `ban-kvqa-fasttext-pkb.pth`)는 다음 경로에 저장합니다.

```
saved_models
├── ban-kvqa-fasttext-pkb
│   └── ban-kvqa-fasttext-pkb.pth
└── ban-kvqa-roberta-base-rnn
    └── ban-kvqa-roberta-base-rnn.pth
```

2. 모델 세팅을 위한 바이너리 파일(`*.vec`, `*.pkl`, `*.npy`)은 다음 경로에 저장합니다.
```
data
├──fasttext/
│   └── ko.vec
├── dictionary_kkma.kvqa.pkl
├── ft_init.kvqa.npy
└── cache
    └── trainval_label2ans.kvqa.pkl
```

### 추론하기

위의 세팅을 마친 후 `ban_kvqa.py`를 실행하면 다음 이미지에 대한 inference test가 실행됩니다.

#### 실행 커맨드
```bash
python3 ban_kvqa.py
```

#### 테스트 이미지
![Image](http://images.cocodataset.org/val2017/000000039769.jpg)

#### 실행 결과
```bash
질문을 입력하세요 # enter로 넘어갈 시, 기본값 `화면에 뭐가 보여?`로 실행
img from url
('고양이', True)
img from PIL
('고양이', True)
```

### 성능
BERT의 경우 `KLUE/robeta-base`를 freeze하여 사용하였습니다. 재현 결과 원본 레포에 근사한 성능을 보여줍니다.
<br />
<br />
#### 원본 레포 성능 보고
| Embedding | Dimension |          All          |  Yes/No   |  Number   |   Other   | Unanswerable |
| --------- | :-------: | :-------------------: | :-------: | :-------: | :-------: | :----------: |
| [fastText](https://arxiv.org/abs/1607.04606)  | [200](https://github.com/Kyubyong/wordvectors)       | **30.94 ± 0.09**  |   72.48   | **17.74** | **18.96** |    77.92     |
| [BERT](https://arxiv.org/abs/1810.04805)      | [768](https://github.com/google-research/bert)       | 30.56 ± 0.12 |   69.28   |   17.48   |   18.65   |    78.28     |

<br />
<br />

#### 재현 성능 보고
각 모델에 대해 학습 중 로깅된 기록은 다음과 같습니다. 하이퍼파라미터값은 `configs/vqa_*.yaml`을 참고하시기 바랍니다.

#### `ban-kvqa-roberta-base-rnn.pth`
- Score : `30.71`
- Inference Time
  - `PIL.Image.Image`로 추론 : `0.492`
  - 이미지 URL로 추론 : `1.451`

<details markdown="1">
<summary>로그</summary>

```
# Score
epoch 19, time: 291.12
	train_loss: 2.36, norm: 3.7897, score: 43.34, confidence: 97.78
	eval score: 30.71 (64.16)
	confidence: 92.67 (100.00)
	entropy:  1.35 3.42 2.89 3.45 3.79 3.52 4.07 4.31

Val upper bound: 0.6416199986457825
Val score: 0.30713999898433686

# Inference Time
(basic) root@030b4173fbf4:~/BAN-KVQA# python inference_vqa.py 

## 1. 모듈 생성 (최초 실행)
The runtime for _init_detector took 6.177962779998779 seconds to complete
The runtime for _init_vqa took 11.176941633224487 seconds to complete

## 2. PIL.Image.Image로부터 추론
The runtime for _init_detector took 1.4332857131958008 seconds to complete
The runtime for _init_vqa took 10.28046727180481 seconds to complete
The runtime for _detect_img took 0.427736759185791 seconds to complete
The runtime for _tokenize_question took 0.0008788108825683594 seconds to complete
The runtime for _preprocess_query took 0.4287285804748535 seconds to complete
The runtime for answer took 0.49230003356933594 seconds to complete
('고양이', True)
__main__.SomeTest.test_inference_from_PILIMAGE: 0.492

## 3. 이미지 URL로부터 추론
The runtime for _init_detector took 1.425206184387207 seconds to complete
The runtime for _init_vqa took 10.267421245574951 seconds to complete
The runtime for _detect_img took 1.3959167003631592 seconds to complete
The runtime for _tokenize_question took 0.0004367828369140625 seconds to complete
The runtime for _preprocess_query took 1.3964643478393555 seconds to complete
The runtime for answer took 1.4507060050964355 seconds to complete
('고양이', True)
__main__.SomeTest.test_inference_from_url: 1.451
----------------------------------------------------------------------
Ran 3 tests in 57.830s
```

</details>

#### `ban-kvqa-fasttext-pkb.pth`
- Score : `30.81`
- Inference Time
  - `PIL.Image.Image`로 추론 : `0.504`
  - 이미지 URL로 추론 : `1.461`
<details markdown="1">
<summary>로그</summary>

```
# Score
epoch 15, time: 200.12
	train_loss: 2.40, norm: 3.5731, score: 42.84, confidence: 97.64
	eval score: 30.81 (64.16)
	confidence: 92.58 (100.00)
	entropy:  0.95 2.40 3.49 3.04 3.15 3.18 4.35 3.95

Val upper bound: 0.6416199986457825
Val score: 0.30825999782085417

# Inference Time
(basic) root@030b4173fbf4:~/BAN-KVQA# python inference_vqa.py 
## 1. 모듈 생성 (최초 실행)
The runtime for _init_detector took 6.160168886184692 seconds to complete
The runtime for _init_vqa took 7.241246223449707 seconds to complete

## 2. PIL.Image.Image로부터 추론
The runtime for _init_detector took 1.624925136566162 seconds to complete
The runtime for _init_vqa took 0.6422216892242432 seconds to complete
The runtime for _detect_img took 0.4179646968841553 seconds to complete
The runtime for _tokenize_question took 0.04199504852294922 seconds to complete
The runtime for _preprocess_query took 0.46011805534362793 seconds to complete
The runtime for answer took 0.5036697387695312 seconds to complete
__main__.SomeTest.test_inference_from_PILIMAGE: 0.504

## 3. 이미지 URL로부터 추론
The runtime for _init_detector took 1.5702142715454102 seconds to complete
The runtime for _init_vqa took 0.986760139465332 seconds to complete
The runtime for _detect_img took 1.3976850509643555 seconds to complete
The runtime for _tokenize_question took 0.020783185958862305 seconds to complete
The runtime for _preprocess_query took 1.418597936630249 seconds to complete
The runtime for answer took 1.4611985683441162 seconds to complete
__main__.SomeTest.test_inference_from_url: 1.461
----------------------------------------------------------------------
Ran 3 tests in 27.560s
```
</details>

### 인용

```
@inproceedings{Kim_Lim2019,
author = {Kim, Jin-hwa and Lim, Soohyun and Park, Jaesun and Cho₩, Hansu},
booktitle = {AI for Social Good workshop at NeurIPS},
title = {{Korean Localization of Visual Question Answering for Blind People}},
year = {2019}
}
@inproceedings{Kim2018,
author = {Kim, Jin-Hwa and Jun, Jaehyun and Zhang, Byoung-Tak},
booktitle = {Advances in Neural Information Processing Systems 31},
title = {{Bilinear Attention Networks}},
pages = {1571--1581},
year = {2018}
}
@misc{park2021klue,
title={KLUE: Korean Language Understanding Evaluation},
author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
year={2021},
eprint={2105.09680},
archivePrefix={arXiv},
primaryClass={cs.CL}
}
```

### 라이센스

* Korean VQA License for the KVQA Dataset
* Creative Commons License Deed ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ko)) for the VizWiz subset
* GNU GPL v3.0 for the Code

