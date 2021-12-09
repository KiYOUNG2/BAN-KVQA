import torch
from typing import Union, List, Tuple
from PIL.Image import Image
class QABase:
    def __init__(self, args):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    def answer(
        self,
        query: str,
        context: Union[Image, Union[str, List[str]]]
    ) -> Tuple[str, bool]: # str : answer, bool : answeralbe or not
        """Return answer when the question is answerable"""
        return NotImplemented