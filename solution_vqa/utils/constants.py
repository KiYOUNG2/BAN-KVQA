import os

# Object Detection Model
VISION_COLUMN_NAME = "v_feature"
SPATIAL_COLUMN_NAME = "spatials"
QUESTION_COLUMN_NAME = "question"
ANSWERABLE_COLUMN_NAME = "answerable"
# ====================================================
DETECTOR_ROOT_PATH = './bottom_up_attention_pytorch/'
DATA_PATH = os.path.join(DETECTOR_ROOT_PATH, 'evaluation')
DETECTOR_CONFIG_FILE_PATH = os.path.join(DETECTOR_ROOT_PATH, 'configs/bua-caffe/extract-bua-caffe-r101.yaml')
OBJECT_VOCAB_FILE = 'objects_vocab.txt'
ATTR_VOCAB_FILE = 'attributes_vocab.txt'
DETECTOR_MIN_BOXES = 10
DETECTOR_MAX_BOXES = 30
DETECTOR_CONF_THRESH = 0.4

# VQA Model
VQA_CONFIG_FILE_PATH = './configs/vqa.yaml'
LABEL2ANS_FILE = './data/cache/trainval_label2ans.kvqa.pkl'
ANS2LABEL_FILE = './data/cache/trainval_ans2label.kvqa.pkl'