import os
import torch
import numpy as np
import cv2
import math
from PIL import Image

from ..model.vocab import Vocab
from ..model.transformerocr import VietOCR


def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    """data: BxCxHxW"""
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token] * len(img)]
        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T == eos_token, axis=1)):
            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = output.to('cpu')
            
            values, indices = torch.topk(output, 1)
            indices = indices[:, -1, 0]
            indices = indices.tolist()

            translated_sentence.append(indices)
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T

    return translated_sentence


def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']
    
    model = VietOCR(len(vocab),
            config['backbone'],
            config['cnn'], 
            config['transformer'],
            config['seq_modeling'])
    
    model = model.to(device)

    return model, vocab

def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w/round_to)*round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height

def process_image(image, image_height, image_min_width, image_max_width):
    img = image.convert('RGB')

    w, h = img.size
    new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = getattr(Image, "LANCZOS", Image.BICUBIC)
    img = img.resize((new_w, image_height), resample=resample)

    img = np.asarray(img).transpose(2,0, 1)
    img = img/255
    return img


def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...]
    img = torch.FloatTensor(img)
    return img


class Predictor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.model, self.vocab = build_model(config)
        self.model.to(self.device)
        self.model.eval()
        weights = config.get("weights")
        if weights:
            self._load_weights(weights)

    def _load_weights(self, weights):
        weights_path = os.path.normpath(weights)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        state = torch.load(weights_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)

    def _prepare_image(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ds_cfg = self.config.get("dataset", {})
        img_h = ds_cfg.get("image_height", 32)
        img_min_w = ds_cfg.get("image_min_width", 32)
        img_max_w = ds_cfg.get("image_max_width", 512)
        return process_input(img, img_h, img_min_w, img_max_w).to(self.device)

    def predict(self, img, max_seq_length=None):
        if max_seq_length is None:
            max_seq_length = self.config.get("transformer", {}).get(
                "max_seq_length",
                128,
            )
        img_tensor = self._prepare_image(img)
        with torch.no_grad():
            seq = translate(
                img_tensor,
                self.model,
                max_seq_length=max_seq_length,
                sos_token=self.vocab.go,
                eos_token=self.vocab.eos,
            )
        return self.vocab.decode(seq[0].tolist())



