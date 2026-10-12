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
import logging
import re
import time

import numpy as np

from ocr_service.core.logging import configure_logging
from ocr_service.engines.layout_recognizer import LayoutRecognizer4YOLOv10 as LayoutRecognizer
from ocr_service.engines.table_structure_recognizer import TableStructureRecognizer
from ocr_service.engines.seeit import draw_box
from ocr_service.services.images import init_in_out

logger = logging.getLogger(__name__)


def main(args):
    images, outputs = init_in_out(args)
    if args.mode.lower() == "layout":
        detr = LayoutRecognizer("layout")
        layouts = detr.forward(images, thr=float(args.threshold))
        for i, lyt in enumerate(layouts):
            logger.info(f"Image {i}:")
            for region in lyt:
                logger.info(f"  Detected label: {region.get('type', '')}")
    if args.mode.lower() == "tsr":
        from ocr_service.engines.ocr_pytorch import OCR
        detr = TableStructureRecognizer()
        ocr = OCR()
        layouts = detr(images, thr=float(args.threshold))
    for i, lyt in enumerate(layouts):
        if args.mode.lower() == "tsr":
            # lyt = [t for t in lyt if t["type"] == "table column"]
            markdown = get_table_markdown(images[i], lyt, ocr)
            with open(outputs[i] + ".md", "w+", encoding='utf-8') as f:
                f.write(markdown)
            lyt = [{
                "type": t["label"],
                "bbox": [t["x0"], t["top"], t["x1"], t["bottom"]],
                "score": t["score"]
            } for t in lyt]
        img = draw_box(images[i], lyt, detr.labels, float(args.threshold))
        img.save(outputs[i], quality=95)
        img.close()
        logging.info("save result to: " + outputs[i])


def get_table_markdown(img, tb_cpns, ocr):
    boxes = ocr(np.array(img))
    boxes = LayoutRecognizer.sort_Y_firstly(
        [{"x0": b[0][0], "x1": b[1][0],
          "top": b[0][1], "text": t[0],
          "bottom": b[-1][1],
          "layout_type": "table",
          "page_number": 0} for b, t in boxes if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]],
        np.mean([b[-1][1] - b[0][1] for b,_ in boxes]) / 3
    )

    def gather(kwd, fzy=10, ption=0.6):
        nonlocal boxes
        eles = LayoutRecognizer.sort_Y_firstly(
            [r for r in tb_cpns if re.match(kwd, r["label"])], fzy)
        eles = LayoutRecognizer.layouts_cleanup(boxes, eles, 5, ption)
        return LayoutRecognizer.sort_Y_firstly(eles, 0)

    headers = gather(r".*header$")
    rows = gather(r".* (row|header)")
    spans = gather(r".*spanning")
    clmns = sorted([r for r in tb_cpns if re.match(
        r"table column$", r["label"])], key=lambda x: x["x0"])
    clmns = LayoutRecognizer.layouts_cleanup(boxes, clmns, 5, 0.5)

    for b in boxes:
        ii = LayoutRecognizer.find_overlapped_with_threashold(b, rows, thr=0.3)
        if ii is not None:
            b["R"] = ii
            b["R_top"] = rows[ii]["top"]
            b["R_bott"] = rows[ii]["bottom"]

        ii = LayoutRecognizer.find_overlapped_with_threashold(b, headers, thr=0.3)
        if ii is not None:
            b["H_top"] = headers[ii]["top"]
            b["H_bott"] = headers[ii]["bottom"]
            b["H_left"] = headers[ii]["x0"]
            b["H_right"] = headers[ii]["x1"]
            b["H"] = ii

        ii = LayoutRecognizer.find_horizontally_tightest_fit(b, clmns)
        if ii is not None:
            b["C"] = ii
            b["C_left"] = clmns[ii]["x0"]
            b["C_right"] = clmns[ii]["x1"]

        ii = LayoutRecognizer.find_overlapped_with_threashold(b, spans, thr=0.3)
        if ii is not None:
            b["H_top"] = spans[ii]["top"]
            b["H_bott"] = spans[ii]["bottom"]
            b["H_left"] = spans[ii]["x0"]
            b["H_right"] = spans[ii]["x1"]
            b["SP"] = ii

    markdown = TableStructureRecognizer.construct_table(boxes, markdown=True)
    return markdown


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',
                        help="Directory, image path, or PDF path",
                        required=True)
    parser.add_argument('--output_dir', help="Output directory. Default: './layouts_outputs'",
                        default="./layouts_outputs")
    parser.add_argument(
        '--threshold',
        help="Detection threshold. Default: 0.5",
        default=0.5)
    parser.add_argument('--mode', help="Layout or table structure recognition", choices=["layout", "tsr"],
                        default="layout")
    args = parser.parse_args(argv)
    configure_logging("recognizer")
    main(args)
