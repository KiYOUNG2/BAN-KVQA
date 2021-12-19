from dataclasses import dataclass
import os

def get_abs_path(base_path, path):
    return os.path.abspath(
        os.path.join(base_path, path)
    )

@dataclass
class Config:
    PROJECT_BASE_PATH:str = os.path.dirname(os.path.dirname(__file__))
    PROJECT_DATA_PATH:str = get_abs_path(PROJECT_BASE_PATH, 'data')

    # Object Detection Model
    VISION_COLUMN_NAME:str = "v_feature"
    SPATIAL_COLUMN_NAME:str = "spatials"
    QUESTION_COLUMN_NAME:str = "question"
    ANSWERABLE_COLUMN_NAME:str = "answerable"
    DETECTOR_MIN_BOXES:int = 10
    DETECTOR_MAX_BOXES:int = 30
    DETECTOR_CONF_THRESH:int = 0.4
    DETECTOR_ROOT_PATH:str = get_abs_path(PROJECT_BASE_PATH, "bottom_up_attention_pytorch/")
    DETECTOR_DATA_PATH:str = os.path.join(DETECTOR_ROOT_PATH, 'evaluation')
    DETECTOR_CONFIG_FILE_PATH:str = os.path.join(DETECTOR_ROOT_PATH, 'configs/bua-caffe/extract-bua-caffe-r101.yaml')
    OBJECT_VOCAB_FILE:str = 'objects_vocab_ko.txt'
    ATTR_VOCAB_FILE:str = 'attributes_vocab_ko.txt'

    # VQA Model
    # VQA_CONFIG_FILE = './configs/vqa_roberta-base-rnn.yaml'
    VQA_CONFIG_FILE:str = get_abs_path(PROJECT_BASE_PATH, 'configs/vqa_fasttext-pkb.yaml')
    VQA_BERT_WEIGHT_FILE:str = get_abs_path(PROJECT_BASE_PATH, 'saved_models/ban-kvqa-roberta-base-rnn/ban-kvqa-roberta-base-rnn.pth')
    VQA_FASTTEXT_WEIGHT_FILE:str = get_abs_path(PROJECT_BASE_PATH, 'saved_models/ban-kvqa-fasttext-pkb/ban-kvqa-fasttext-pkb.pth')
    LABEL2ANS_FILE:str = get_abs_path(PROJECT_BASE_PATH, 'data/cache/trainval_label2ans.kvqa.pkl')
    ANS2LABEL_FILE:str = get_abs_path(PROJECT_BASE_PATH, 'data/cache/trainval_ans2label.kvqa.pkl')