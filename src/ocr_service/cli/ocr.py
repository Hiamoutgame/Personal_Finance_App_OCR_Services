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

import argparse
import json
import logging
import time

from ocr_service.core.logging import configure_logging
from ocr_service.engines.seeit import draw_box
from ocr_service.services.images import init_in_out
from ocr_service.services.invoice import boxes_to_text, ocr_image_to_boxes, ocr_image_to_invoice_layout, _write_text_json

logger = logging.getLogger(__name__)


def main(args):
    from ocr_service.engines.ocr_pytorch import OCR
    ocr = OCR()
    images, outputs = init_in_out(args)

    def __ocr(i, device_id, img):
        logger.info("Task %d start", i)
        start_time = time.time()
        boxes = ocr_image_to_boxes(ocr, img, device_id)
        boxed_img = draw_box(images[i].copy(), boxes, ["ocr"], 1.0)
        boxed_img.save(outputs[i], quality=95)
        text = boxes_to_text(boxes)
        with open(outputs[i] + ".txt", "w+", encoding="utf-8") as f:
            f.write(text)
        _write_text_json(outputs[i] + ".text.json", text)
        if args.layout == "invoice":
            invoice_layout = ocr_image_to_invoice_layout(ocr, img, device_id, boxes=boxes)
            with open(outputs[i] + ".json", "w+", encoding="utf-8") as f:
                json.dump(invoice_layout, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time
        logger.info("Task %d done in %.2f seconds", i, elapsed)

    for index, image in enumerate(images):
        try:
            __ocr(index, 0, image)
        finally:
            image.close()
    logger.info("OCR completed!")


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        help="Directory, PDF path, image path, or a single image/PDF file path.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Directory for output images and text files. Default: './ocr_outputs'",
        default="./ocr_outputs",
    )
    parser.add_argument(
        "--layout",
        choices=["none", "invoice"],
        default="invoice",
        help="Structured layout output mode. Use 'invoice' to write a .json layout next to the .txt file.",
    )
    cli_args = parser.parse_args(argv)
    configure_logging("ocr")
    main(cli_args)
