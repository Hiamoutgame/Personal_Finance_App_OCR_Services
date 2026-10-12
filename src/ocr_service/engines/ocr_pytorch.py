#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import cv2
import numpy as np
from PIL import Image

from ocr_service.core.paths import onnx_directory, vietocr_directory
from ocr_service.vendor.vietocr.tool.config import Cfg
from ocr_service.vendor.vietocr.tool.translate import Predictor
from .ocr_onnx import OCR as OnnxOCR, TextDetector


class TextRecognizer:
    def __init__(self):
        config = Cfg.load_config_from_name("vgg-seq2seq")
        config["weights"] = vietocr_directory("weight", "vgg_seq2seq.pth")
        config["cnn"]["pretrained"] = False
        config["device"] = "cpu"
        self.detector = Predictor(config)

    def __call__(self, img_list):
        results = []
        for image in img_list:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            results.append((self.detector.predict(image), 1.0))
        return results, 0.0


class OCR(OnnxOCR):
    def __init__(self, model_dir=None):
        self.text_detector = [TextDetector(model_dir or onnx_directory(), 0)]
        self.text_recognizer = [TextRecognizer()]
        self.drop_score = 0.5
        self.crop_image_res_index = 0
