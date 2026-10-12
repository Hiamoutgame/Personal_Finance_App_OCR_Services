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


def extract_table_markdown(img, table_region, ocr):
    # Use bbox if present
    if "bbox" in table_region:
        x0, y0, x1, y1 = map(int, table_region["bbox"])
    else:
        x0, y0, x1, y1 = map(int, [table_region["x0"], table_region["top"], table_region["x1"], table_region["bottom"]])
    table_img = img.crop((x0, y0, x1, y1))
    tb_cpns = TableStructureRecognizer()([table_img])[0]
    boxes = ocr(np.array(table_img))
    boxes = LayoutRecognizer.sort_Y_firstly(
        [{"x0": b[0][0], "x1": b[1][0],
          "top": b[0][1], "text": t[0],
          "bottom": b[-1][1],
          "layout_type": "table",
          "page_number": 0} for b, t in boxes if b[0][0] <= b[1][0] and b[0][1] <= b[-1][1]],
        np.mean([b[-1][1] - b[0][1] for b, _ in boxes]) / 3
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

def main(args):
    from ocr_service.engines.ocr_pytorch import OCR
    images, outputs = init_in_out(args)
    logger.info(f"Loaded {len(images)} images")
    logger.info(f"Output paths: {outputs}")
    layout_recognizer = LayoutRecognizer("layout")
    ocr = OCR()
    for idx, img in enumerate(images):
        logger.info(f"Processing image {idx}: {outputs[idx]}")
        start_time = time.time()  # <-- Start timing

        layouts = layout_recognizer.forward([img], thr=float(args.threshold))[0]
        logger.info(f"Detected {len(layouts)} layout regions")
        region_and_pos = []

        from PIL import Image, ImageDraw

        # Create a mask for detected regions
        mask = Image.new("1", img.size, 0)
        draw = ImageDraw.Draw(mask)
        for region in layouts:
            if "bbox" in region:
                x0, y0, x1, y1 = map(int, region["bbox"])
            else:
                x0, y0, x1, y1 = map(int, [region.get("x0", 0), region.get("top", 0), region.get("x1", 0), region.get("bottom", 0)])
            draw.rectangle([x0, y0, x1, y1], fill=1)

        for region in layouts:
            label = region.get("type", "").lower()
            score = region.get("score", 1.0)
            bbox = region.get("bbox", [region.get("x0", 0), region.get("top", 0), region.get("x1", 0), region.get("bottom", 0)])
            y_pos = bbox[1]  # Use top y as position for ordering
            if label in ["table"] and score >= float(args.threshold):
                logger.info(f"Extracting table markdown for region: {region}")
                markdown = extract_table_markdown(img, region, ocr)
                region_and_pos.append((y_pos, markdown))

        # Now OCR any remaining undetected area (including non-table/figure)
        inv_mask = mask.point(lambda p: 1 - p)
        if inv_mask.getbbox():
            x0, y0, x1, y1 = inv_mask.getbbox()
            region_img = img.crop((x0, y0, x1, y1))
            ocr_results = ocr(np.array(region_img))
            text = "\n".join([t[0] for _, t in ocr_results if t and t[0]])
            region_and_pos.append((y0, text))

        # Sort by y position to preserve original order
        region_and_pos.sort(key=lambda x: x[0])
        markdown_concat = "\n\n".join([item[1] for item in region_and_pos])
        out_path = outputs[idx] + "_full.md"
        logger.info(f"Writing concatenated markdown to: {out_path}")
        with open(out_path, "w+", encoding='utf-8') as f:
            f.write(markdown_concat)
        logging.info(f"Saved concatenated markdown to: {out_path}")

        elapsed = time.time() - start_time  # <-- End timing
        logger.info("Processing image %d done in %.2f seconds", idx, elapsed)
        img.close()

def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs',
                        help="Directory or file path for images or PDFs",
                        required=True)
    parser.add_argument('--output_dir', help="Directory for output markdown files. Default: './table_markdown_outputs'",
                        default="./table_markdown_outputs")
    parser.add_argument('--threshold',
                        help="Detection threshold. Default: 0.5",
                        default=0.5)
    args = parser.parse_args(argv)
    configure_logging("pipeline")
    main(args)
