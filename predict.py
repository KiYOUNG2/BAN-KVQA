import io
import streamlit as st
import torch
from PIL import Image
from ban_kvqa import VQA
from typing import Tuple

@st.cache
def load_model() -> VQA:
    model = VQA()
    return model
    
def get_prediction(model:VQA, query: str, image_bytes: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(io.BytesIO(image_bytes))
    pred, answerable = model.answer(query, image)
    
    if answerable:
        return pred
    else:
        return "Unanswerable"