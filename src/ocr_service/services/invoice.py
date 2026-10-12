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

import json
import re

import numpy as np

from .text_format import _norm_text


INVOICE_LAYOUT_CONFIG = {
    "type": "invoice",
    "sections": {
        "header": {
            "description": "Store/vendor, title, address, contact, invoice metadata.",
        },
        "items": {
            "description": "Purchased goods/services, quantities, unit prices, and line totals.",
        },
        "totals": {
            "description": "Subtotal, tax, fees, total, payment, and remaining amount.",
            "keywords": (
                "tong",
                "tong cong",
                "thanh toan",
                "khach dua",
                "no lai",
                "tien nuoc",
                "thue",
                "vat",
                "gtgt",
                "phi",
                "phu phi",
                "tieu thu",
            ),
        },
        "footer": {
            "description": "Cashier/staff, thanks text, print time, website, and hotline.",
            "keywords": (
                "cam on",
                "nhan vien",
                "dien thoai",
                "ngay gui",
                "in luc",
                "hotline",
                "www",
            ),
        },
    },
    "item_header_keywords": (
        "description",
        "dien giai",
        "ten hang",
        "mat hang",
        "don gia",
        "so luong",
        "qty",
    ),
}


def ocr_image_to_boxes(ocr, image, device_id=0):
    results = ocr(np.array(image), device_id)
    boxes = [(line[0], line[1][0]) for line in results]
    return [
        {
            "text": text,
            "bbox": [box[0][0], box[0][1], box[1][0], box[-1][1]],
            "type": "ocr",
            "score": 1,
        }
        for box, text in boxes
        if box[0][0] <= box[1][0] and box[0][1] <= box[-1][1]
    ]


def boxes_to_text(boxes):
    return "\n".join(box["text"] for box in boxes if box.get("text"))


def ocr_image_to_text(ocr, image, device_id=0):
    return boxes_to_text(ocr_image_to_boxes(ocr, image, device_id))


def _write_text_json(path, text):
    with open(path, "w+", encoding="utf-8") as f:
        json.dump({"text": text}, f, ensure_ascii=False, indent=2)


def _has_keyword(normalized_text, keyword):
    keyword = _norm_text(keyword)
    if not keyword:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def _box_center_y(box):
    x0, y0, x1, y1 = box["bbox"]
    return (y0 + y1) / 2


def _union_bbox(items):
    bboxes = [item["bbox"] for item in items if item.get("bbox")]
    if not bboxes:
        return None
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def group_boxes_into_lines(boxes):
    if not boxes:
        return []

    heights = [box["bbox"][3] - box["bbox"][1] for box in boxes if box.get("bbox")]
    line_threshold = max(8, float(np.median(heights)) * 0.65) if heights else 12
    lines = []

    for box in sorted(boxes, key=lambda item: (_box_center_y(item), item["bbox"][0])):
        cy = _box_center_y(box)
        matched = None
        for line in lines:
            if abs(cy - line["center_y"]) <= line_threshold:
                matched = line
                break

        if matched is None:
            matched = {"boxes": [], "center_y": cy}
            lines.append(matched)

        matched["boxes"].append(box)
        matched["center_y"] = float(np.mean([_box_center_y(b) for b in matched["boxes"]]))

    result = []
    for index, line in enumerate(sorted(lines, key=lambda item: item["center_y"])):
        line_boxes = sorted(line["boxes"], key=lambda item: item["bbox"][0])
        text = " ".join(box["text"] for box in line_boxes if box.get("text"))
        result.append(
            {
                "index": index,
                "text": text,
                "bbox": _union_bbox(line_boxes),
                "boxes": line_boxes,
            }
        )
    return result


def _line_has_any(line, keywords):
    text = _norm_text(line.get("text", ""))
    return any(_has_keyword(text, keyword) for keyword in keywords)


def _amount_candidates(text):
    candidates = []
    amount_pattern = r"(?<!\d)(\d{1,3}(?:[.,]\d{3})+|\d{4,}\s*(?:vnd|d|đ))(?!\d)"
    for match in re.finditer(amount_pattern, text, flags=re.IGNORECASE):
        raw = match.group(0).strip()
        digits = re.sub(r"\D", "", raw)
        if not digits:
            continue
        candidates.append(
            {
                "raw": raw,
                "value": int(digits),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return candidates


def _best_amount(text):
    candidates = _amount_candidates(text)
    if not candidates:
        return None
    return candidates[-1]


def _find_dates(text):
    return re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)


def _find_phones(text):
    phones = []
    for match in re.finditer(r"(?<!\d)(0\d(?:[\s.]*\d){7,10})(?!\d)", text):
        phone = re.sub(r"\D", "", match.group(1))
        if len(phone) >= 9:
            phones.append(phone)
    return phones


def _section_payload(lines):
    return {
        "text": "\n".join(line["text"] for line in lines if line.get("text")),
        "bbox": _union_bbox(lines),
        "lines": lines,
    }


def split_invoice_sections(lines, image_size):
    if not lines:
        return {name: _section_payload([]) for name in INVOICE_LAYOUT_CONFIG["sections"]}

    totals_keywords = INVOICE_LAYOUT_CONFIG["sections"]["totals"]["keywords"]
    strong_totals_keywords = (
        "tong",
        "tong cong",
        "thanh toan",
        "khach dua",
        "no lai",
        "tien nuoc",
        "tieu thu",
    )
    footer_keywords = INVOICE_LAYOUT_CONFIG["sections"]["footer"]["keywords"]
    item_header_keywords = INVOICE_LAYOUT_CONFIG["item_header_keywords"]

    def is_totals_line(line):
        normalized = _norm_text(line.get("text", ""))
        if any(_has_keyword(normalized, keyword) for keyword in strong_totals_keywords):
            return True
        return _best_amount(line.get("text", "")) is not None and any(
            _has_keyword(normalized, keyword) for keyword in totals_keywords
        )

    item_header_idx = next(
        (i for i, line in enumerate(lines) if _line_has_any(line, item_header_keywords)),
        None,
    )
    totals_idx = next(
        (i for i, line in enumerate(lines) if is_totals_line(line)),
        None,
    )
    footer_idx = None
    if totals_idx is not None:
        footer_idx = next(
            (i for i, line in enumerate(lines[totals_idx + 1 :], start=totals_idx + 1) if _line_has_any(line, footer_keywords)),
            None,
        )

    if item_header_idx is not None:
        item_start = item_header_idx + 1
    else:
        min_item_y = image_size[1] * 0.18 if image_size else 0
        item_start = next(
            (
                i
                for i, line in enumerate(lines)
                if i > 3
                and line.get("bbox")
                and line["bbox"][1] >= min_item_y
                and _best_amount(line["text"])
                and not _line_has_any(line, totals_keywords)
            ),
            None,
        )

    if totals_idx is None and image_size:
        height = image_size[1]
        totals_idx = next(
            (
                i
                for i, line in enumerate(lines)
                if line.get("bbox") and line["bbox"][1] > height * 0.58 and _best_amount(line["text"])
            ),
            None,
        )

    header_end = item_start if item_start is not None else (totals_idx if totals_idx is not None else len(lines))
    items_end = totals_idx if totals_idx is not None else len(lines)
    totals_end = footer_idx if footer_idx is not None else len(lines)

    sections = {
        "header": lines[:header_end],
        "items": lines[item_start:items_end] if item_start is not None else [],
        "totals": lines[totals_idx:totals_end] if totals_idx is not None else [],
        "footer": lines[footer_idx:] if footer_idx is not None else [],
    }
    return {name: _section_payload(section_lines) for name, section_lines in sections.items()}


def extract_invoice_fields(lines):
    full_text = "\n".join(line["text"] for line in lines)
    normalized_lines = [(_norm_text(line["text"]), line) for line in lines]
    fields = {
        "dates": _find_dates(full_text),
        "phones": _find_phones(full_text),
    }

    tax_code = re.search(r"(?:ma\s*so\s*thue|mst)\D*([0-9\s.-]{8,20})", _norm_text(full_text))
    if tax_code:
        fields["tax_code"] = re.sub(r"\D", "", tax_code.group(1))

    amount_fields = {
        "total_amount": ("thanh toan", "tong cong", "tong:", "tong "),
        "subtotal": ("tam tinh", "tien hang", "tien nuoc"),
        "tax": ("thue", "vat", "gtgt"),
        "fee": ("phi", "phu phi", "bvmt"),
        "customer_paid": ("khach dua",),
        "remaining": ("no lai",),
        "consumption": ("tieu thu",),
    }
    for field_name, keywords in amount_fields.items():
        matches = []
        for index, (normalized, line) in enumerate(normalized_lines):
            if not any(_has_keyword(normalized, keyword) for keyword in keywords):
                continue
            amount = _best_amount(line["text"])
            if field_name == "total_amount":
                nearby_amounts = []
                if amount:
                    nearby_amounts.append((amount, line))
                for _, nearby_line in normalized_lines[index + 1 : index + 3]:
                    nearby_amount = _best_amount(nearby_line["text"])
                    if nearby_amount:
                        nearby_amounts.append((nearby_amount, nearby_line))
                if nearby_amounts:
                    amount, amount_line = max(nearby_amounts, key=lambda item: item[0]["value"])
                    matches.append(
                        {
                            "value": amount["value"],
                            "raw": amount["raw"],
                            "line": amount_line["text"],
                            "bbox": amount_line["bbox"],
                        }
                    )
                continue
            if amount:
                matches.append(
                    {
                        "value": amount["value"],
                        "raw": amount["raw"],
                        "line": line["text"],
                        "bbox": line["bbox"],
                    }
                )
        if matches:
            fields[field_name] = matches[-1] if field_name == "total_amount" else matches

    return fields


def extract_invoice_items(item_lines):
    items = []
    pending_name = []

    for line in item_lines:
        text = line["text"].strip()
        normalized = _norm_text(text)
        if not text or set(text) <= {"-", "_", ".", " "}:
            continue
        if _line_has_any(line, INVOICE_LAYOUT_CONFIG["item_header_keywords"]):
            continue

        amount = _best_amount(text)
        if not amount:
            pending_name.append(text)
            continue

        name_part = text[: amount["start"]].strip(" -:|")
        qty_match = re.search(
            r"(?<!\d)(\d+(?:[.,]\d{1,3})?)\s*(?:kg|bic|goi|gói|lon|chai|cai|cái|vi|vỉ|hu|hũ|m3)?\s*x",
            normalized,
        )
        quantity = qty_match.group(1).replace(",", ".") if qty_match else None
        unit_price = None
        all_amounts = _amount_candidates(text)
        if len(all_amounts) >= 2:
            unit_price = all_amounts[-2]["value"]

        name = " ".join(pending_name + ([name_part] if name_part else [])).strip()
        name = re.sub(r"^\d+\s*[\).:-]?\s*", "", name).strip()
        pending_name = []

        if not name:
            name = text[: amount["start"]].strip()

        items.append(
            {
                "name": name,
                "quantity": quantity,
                "unit_price": unit_price,
                "line_total": amount["value"],
                "raw_text": text,
                "bbox": line["bbox"],
            }
        )

    return items


def ocr_image_to_invoice_layout(ocr, image, device_id=0, boxes=None):
    if boxes is None:
        boxes = ocr_image_to_boxes(ocr, image, device_id)
    lines = group_boxes_into_lines(boxes)
    sections = split_invoice_sections(lines, image.size)
    return {
        "layout_config": INVOICE_LAYOUT_CONFIG,
        "image": {
            "width": image.size[0],
            "height": image.size[1],
        },
        "text": boxes_to_text(boxes),
        "fields": extract_invoice_fields(lines),
        "sections": sections,
        "items": extract_invoice_items(sections["items"]["lines"]),
        "boxes": boxes,
    }
