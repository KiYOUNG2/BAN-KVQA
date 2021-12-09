# 한국어 시각 질의응답을 위한 Bilinear Attention Networks (BAN) DEMO
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.1.0](https://img.shields.io/badge/pytorch-1.1.0-green.svg?style=plastic)
![cuDNN 7.5](https://img.shields.io/badge/cudnn-7.5-green.svg?style=plastic)

이 코드 저장소는 [SKTBrain/BAN-KVQA](https://github.com/SKTBrain/BAN-KVQA)의 비공식 Inference Pipeline 데모를 제공합니다.
- Huggingface 현재 버전(`v4.12.x`)에 맞게 업데이트
- arguments 입력 방식 변경 : `argparse` 모듈 대신 `configs/` 경로 내 `*.yaml` 파일로 입력
- Image로부터 Inference를 하기 위해 `bottom-up-attention` submodule 추가


![Overview of bilinear attention networks](docs/assets/img/ban_overview.png)

이 코드 저장소의 원본 코드는 @jnhwkim의 [SKTBrain/BAN-KVQA](https://github.com/SKTBrain/BAN-KVQA)임을 밝힙니다. 코드 사용을 허락해주신 김진화님께 감사드립니다.


### 미리 준비할 사항

#### mecab 설치
mecab 설치를 위해서 다음 명령어를 실행하십시오.
```bash
sudo apt-get install default-jre curl
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

#### submodule 세팅(bottom-up-attention)
inference demo를 위해 bottom-up-attention을 사용합니다. 기존 caffe model을 pytorch로 변환한 [MILVLG/bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch)를 submodule로 사용합니다. 이 레포를 클론한 경로에서 다음 순서대로 실행합니다.
```
# Setup Detectron2
cd bottom_up_attention_pytorch
cd detectron2
pip install -e .
cd ..
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install
cd ..
python setup.py build develop
pip install ray
pip install opencv-python scikit-image

# Download object detection model pretrained weights
wget https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EaXvCC3WjtlLvvEfLr3oa8UBLA21tcLh4L8YLbYXl6jgjg?download=1
mv EaXvCC3WjtlLvvEfLr3oa8UBLA21tcLh4L8YLbYXl6jgjg?download=1 bua-caffe-frcn-r101_with_attributes.pth
```


### 데모를 위한 데이터셋 내려받기

- VQA 모델의 pretrained weight와 모델 세팅을 위한 바이너리 파일은 [여기](https://drive.google.com/drive/u/0/folders/1qq_mh3HBCe0WALvIjbD4XnTmgjgjfVNH)에서 다운로드 받으실 수 있습니다.

1. pretrained weight 파일(`ban-kvqa-roberta-base-rnn.pth`)는 다음 경로에 저장합니다.

```bash
saved_models
└── ban-kvqa-roberta-base-rnn
    └── ban-kvqa-roberta-base-rnn.pth
```

2. 모델 세팅을 위한 바이너리 파일(`trainval_label2ans.kvqa.pkl`)은 다음 경로에 저장합니다.
```bash
data
├── KVQA_annotations_train.json
├── KVQA_annotations_val.json
├── KVQA_annotations_test.json
└── cache
    └── trainval_label2ans.kvqa.pkl
```


### 추론하기

위의 세팅을 마친 후 다음 코드를 실행하면 다음 이미지에 대한 inference test가 실행됩니다.

#### 테스트 이미지
![Image](http://images.cocodataset.org/val2017/000000039769.jpg)

#### 실행 커맨드
```bash
python3 inference_vqa.py
```
#### 실행 결과
```bash
질문을 입력하세요 # enter로 넘어갈 시, 기본값 `화면에 뭐가 보여?`로 실행
img from url
('고양이', True)
img from PIL
('고양이', True)
```


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
```

### 라이센스

* Korean VQA License for the KVQA Dataset
* Creative Commons License Deed ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ko)) for the VizWiz subset
* GNU GPL v3.0 for the Code

