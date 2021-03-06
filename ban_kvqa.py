import os
import pickle
import requests
import sys
from collections import OrderedDict
from typing import Any, Iterable, List, NewType, Optional, Tuple, Union, Dict, Callable

import numpy as np
import torch
from transformers import AutoTokenizer

# Object Detection Model
import cv2
from PIL import Image
sys.path.append('./bottom_up_attention_pytorch/detectron2')

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

sys.path.append('./bottom_up_attention_pytorch/')
from utils.extract_utils import get_image_blob
from models.bua import add_bottom_up_attention_config
from models.bua.layers.nms import nms

# Vision-Language Model
from konlpy.tag import Kkma

from solution_vqa.model import base_model
from solution_vqa.utils import dictionary_dict
from solution_vqa.utils import constants as C
from solution_vqa.data.dataset import Dictionary

from args import (
    HfArgumentParser,
    MrcDataArguments,
    MrcModelArguments,
    MrcTrainingArguments,
    MrcProjectArguments,
)


class QABase:
    def __init__(self, args):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
    def answer(
        self,
        query: str,
        context: Union[Image.Image, Union[str, List[str]]]
    ) -> Tuple[str, bool]: # str : answer, bool : answeralbe or not
        """Return answer when the question is answerable"""
        return NotImplemented


class VQA(QABase):
    def __init__(
        self,
    ):
        super().__init__(args=None)
        self._init_detector()
        self._init_vqa()
    def answer(
        self,
        query: str,
        img: Union[str, Image.Image],
    ) -> Tuple[str, bool]: # str : answer, bool : answeralbe or not
        """Return answer when the question is answerable"""

        sample = self._preprocess_query(query, img)

        with torch.no_grad():
            v_features = sample[C.VISION_COLUMN_NAME].to(self.device)
            spatials   = sample[C.SPATIAL_COLUMN_NAME].to(self.device)
            question   = sample[C.QUESTION_COLUMN_NAME].to(self.device)
            pred, _, _ = self.vqa_model(v_features, spatials, question, 'sample', None)
            idx = torch.argmax(pred).item()
            pred_class = self.vqa_label2ans[idx]
        

        if pred_class == 'unanswerable':
            return pred_class, False
        else:
            return pred_class, True

    def _init_detector(self):
        """Initialize Detector Model"""
        # Load Classes & Attribute
        # Load classes
        self.detecter_classes = ['__background__']
        with open(os.path.join(C.DATA_PATH, C.OBJECT_VOCAB_FILE)) as f:
            for object in f.readlines():
                self.detecter_classes.append(object.split(',')[0].lower().strip())

        # Load attributes
        self.detecter_attributes = ['__no_attribute__']
        with open(os.path.join(C.DATA_PATH, C.ATTR_VOCAB_FILE)) as f:
            for att in f.readlines():
                self.detecter_attributes.append(att.split(',')[0].lower().strip())

        # Load Model
        self.detecter_cfg = get_cfg()
        add_bottom_up_attention_config(self.detecter_cfg, True)
        self.detecter_cfg.merge_from_file(C.DETECTOR_CONFIG_FILE_PATH)
        self.detecter_cfg.freeze()

        self.detector_model = DefaultTrainer.build_model(self.detecter_cfg)
        DetectionCheckpointer(self.detector_model, save_dir=self.detecter_cfg.OUTPUT_DIR).resume_or_load(
            os.path.join(C.DETECTOR_ROOT_PATH + self.detecter_cfg.MODEL.WEIGHTS), resume=True
        )
        self.detector_model.eval()

    def _read_img(self, img: Union[str, Image.Image]):
        """Read image from url and convert it to np.array

        Args:
            img (Union[str, Image.Image]): image url string or PIL.Image.Image object
        """

        if type(img).__mro__[-2] == Image.Image:
            return cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)

        elif type(img) == str:
            return cv2.imdecode(
                np.asarray(
                    bytearray(requests.get(img).content),
                    dtype=np.uint8
                    ),
                cv2.IMREAD_COLOR
                )
    def _detect_img(self, img: Union[str, Image.Image]):
        """Detect Object from input image url

        Args:
            img (Union[str, Image.Image]): image url or PIL.Image.Image object
        """
        im = self._read_img(img)
        dataset_dict = get_image_blob(im, self.detecter_cfg.MODEL.PIXEL_MEAN)

        with torch.set_grad_enabled(False):
            boxes, scores, features_pooled, attr_scores = self.detector_model([dataset_dict])

        dets = boxes[0].tensor.cpu() / dataset_dict['im_scale']

        scores = scores[0].cpu()
        feats = features_pooled[0].cpu()
        attr_scores = attr_scores[0].cpu()

        max_conf = torch.zeros((scores.shape[0]), device=scores.device)
        for cls_ind in range(1, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.3)
                max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                            cls_scores[keep],
                                            max_conf[keep])

        keep_boxes = torch.nonzero(max_conf >= C.DETECTOR_CONF_THRESH).flatten()
        if len(keep_boxes) < C.DETECTOR_MIN_BOXES:
            keep_boxes = torch.argsort(max_conf, descending=True)[:C.DETECTOR_MIN_BOXES]
        elif len(keep_boxes) > C.DETECTOR_MAX_BOXES:
            keep_boxes = torch.argsort(max_conf, descending=True)[:C.DETECTOR_MAX_BOXES]

        image_feat = feats[keep_boxes]
        image_bboxes = dets[keep_boxes]

        return {
            'x':image_feat,
            'bbox':image_bboxes,
            'num_bbox':len(keep_boxes),
            'image_h':np.size(im, 0),
            'image_w':np.size(im, 1),
            }
    def _init_vqa(self):
        
        # setup configs
        parser = HfArgumentParser(
            [
                MrcDataArguments,
                MrcModelArguments,
                MrcTrainingArguments,
                MrcProjectArguments
            ]
        )
        args = parser.parse_yaml_file(yaml_file=os.path.abspath(C.VQA_CONFIG_FILE))
        data_args, model_args, _, _ = args

        # load answer label dict
        with open(C.LABEL2ANS_FILE, 'rb') as f:
            self.vqa_label2ans = pickle.load(f)
            # convert list to dict for speed
            self.vqa_label2ans = {k:v for k, v in enumerate(self.vqa_label2ans)}

        # Load dictionary file
        if 'bert' in model_args.architectures:
            self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
            self.dictionary = self.tokenizer.vocab
            NTOKEN = None
            VQA_WEIGHT_FILE = C.VQA_BERT_WEIGHT_FILE

        elif 'fasttext-pkb' in model_args.architectures:
            dictionary_path = os.path.join(
                                    data_args.dataset_path,
                                    dictionary_dict[model_args.architectures]['dict']
                                    )
            self.dictionary = Dictionary.load_from_file(dictionary_path)
            self.tokenizer = Kkma()

            # To speed up Kkma tokenizer, Do tokenize "Dummy input"
            self.tokenizer.morphs("Dummy input")

            NTOKEN = self.dictionary.ntoken
            VQA_WEIGHT_FILE = C.VQA_FASTTEXT_WEIGHT_FILE

        else:
            raise ValueError("Invaild ModelArguments `architectures` argument type: bert, bertrnn, fasttext-pkb")

        # Built Model
        self.vqa_model = getattr(base_model, 'build_ban')(
                            model_args.num_classes,
                            model_args.v_dim,
                            model_args.num_hid,
                            NTOKEN,
                            model_args.op,
                            model_args.gamma,
                            model_args.architectures,
                            model_args.on_do_q,
                            model_args.finetune_q
                        ).to(self.device)

        if 'bert' not in model_args.architectures:
            self.vqa_model.q_emb.w_emb.init_embedding(os.path.join(
                                    data_args.dataset_path,
                                    dictionary_dict[model_args.architectures]['embedding']
                                    )
                                    )

        # Load Checkpoint
        model_data = torch.load(VQA_WEIGHT_FILE)

        model_state = OrderedDict()
        for k, v in model_data.get('model_state', model_data).items():
            model_state[k.replace('module.', '')] = v
        self.vqa_model.load_state_dict(model_state)
        del model_state
        self.vqa_model.eval()
    def _tokenize_question(self, question:str, max_length=16) -> torch.Tensor:
        """Tokenizes the questions.

        Args:
            question (str): User's question
            max_length (int, optional): Max token sequence length. Defaults to 16 for bert, 14 for fasttext

        Returns:
            torch.Tensor: Token sequence(shape : (1, 16) or (1, 14)) 
        """

        if hasattr(self.tokenizer, 'tokenize'):
            return self.tokenizer.encode(
                question,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt',
                add_special_tokens=True
                )
        elif hasattr(self.tokenizer, 'morphs'):
            # max_length -= 2
            tokens = self.tokenizer.morphs(question.replace('.', ''))
            tokens = [self.dictionary.word2idx[token] for token in tokens[:max_length]]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            return torch.tensor([tokens])

    def _preprocess_query(self, question:str, img:Union[str, Image.Image]):
        """Preprocess question and image for VQA Model

        Args:
            question (str): question about input image
            img (Image.Image): image url or PIL.Image.Image object
        """

        img_features = self._detect_img(img)
        question = self._tokenize_question(question)
        return {
            "v_feature"   : img_features['x'].unsqueeze(0),
            "spatials"    : img_features['bbox'].unsqueeze(0),
            "question"    : question,
            }


if __name__ == '__main__':
    vqa = VQA()
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    image.show()
    query = input("????????? ???????????????") or "????????? ?????? ???????"
    # inference from img_url
    print(vqa.answer(query, image_url))

    # inference from PIL
    print(vqa.answer(query, image))