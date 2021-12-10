import os

PROJECT_BASE_PATH = "vqa"

# Object Detection Model
VISION_COLUMN_NAME = "v_feature"
SPATIAL_COLUMN_NAME = "spatials"
QUESTION_COLUMN_NAME = "question"
ANSWERABLE_COLUMN_NAME = "answerable"
# ====================================================

def get_abs_path(base_path, path):
    return os.path.abspath(
        os.path.join(base_path, path)
    )

DETECTOR_ROOT_PATH = get_abs_path(PROJECT_BASE_PATH, "bottom_up_attention_pytorch/")
DATA_PATH = os.path.join(DETECTOR_ROOT_PATH, 'evaluation')
DETECTOR_CONFIG_FILE_PATH = os.path.join(DETECTOR_ROOT_PATH, 'configs/bua-caffe/extract-bua-caffe-r101.yaml')
OBJECT_VOCAB_FILE = 'objects_vocab.txt'
ATTR_VOCAB_FILE = 'attributes_vocab.txt'
DETECTOR_MIN_BOXES = 10
DETECTOR_MAX_BOXES = 30
DETECTOR_CONF_THRESH = 0.4

# VQA Model
# VQA_CONFIG_FILE = './configs/vqa_roberta-base-rnn.yaml'
VQA_CONFIG_FILE = get_abs_path(PROJECT_BASE_PATH, 'configs/vqa_fasttext-pkb.yaml')
VQA_BERT_WEIGHT_FILE = get_abs_path(PROJECT_BASE_PATH, 'saved_models/ban-kvqa-roberta-base-rnn/ban-kvqa-roberta-base-rnn.pth')
VQA_FASTTEXT_WEIGHT_FILE = get_abs_path(PROJECT_BASE_PATH, 'saved_models/ban-kvqa-fasttext-pkb/ban-kvqa-fasttext-pkb.pth')
LABEL2ANS_FILE = get_abs_path(PROJECT_BASE_PATH, 'data/cache/trainval_label2ans.kvqa.pkl')
ANS2LABEL_FILE = get_abs_path(PROJECT_BASE_PATH, 'data/cache/trainval_ans2label.kvqa.pkl')