# Tài liệu này dùng để hướng dẫn kết nối services cho C# BE

Tài liệu này mô tả cách chạy service OCR và cách gọi API /ocr.

## Cách chạy

1. Cài dependencies

```bash
pip install -r requirements.txt
```

2. Chạy service

```bash
python ocr_api.py
```

Tùy chọn host/port (nếu cần):

```bash
python ocr_api.py --host 0.0.0.0 --port 9380
```

3. Mở giao diện Swagger

```
http://<host>:<port>/docs
```

Lưu ý: port mặc định lấy từ config (ragflow) hoặc ENV; nếu không có thì dùng 9380.

## Input nhận vào

**Endpoint:** `POST /ocr`

**Content-Type:** `multipart/form-data`

**Trường bắt buộc:**

- `file`: ảnh hoặc PDF cần OCR

**Query params:**

- `layout` (tùy chọn):
  - `none` (mặc định): chỉ trả `text`
  - `invoice`: trả thêm layout hóa đơn (sections/items/fields/boxes)
  - `document`: trả thêm layout tổng quát theo ONNX (vùng text/table/figure...)

## Output trả ra

**Dạng cơ bản:**

```json
{
  "text": "...",
  "layout": "none|invoice|document",
  "pages": [
    {
      "page": 1,
      "text": "..."
    }
  ]
}
```

**Khi `layout=invoice`:**

- `pages[].invoice` có `sections`, `items`, `fields`, `boxes`
- Có shortcut `invoice` ở cấp gốc nếu chỉ 1 trang

**Khi `layout=document`:**

- `pages[].layout_regions` là danh sách vùng layout
- Có shortcut `layout_regions` ở cấp gốc nếu chỉ 1 trang

**Ghi chú:**

- PDF nhiều trang sẽ ghép `text` bằng dòng trống giữa các trang.

## Ví dụ gọi API

```bash
curl -X POST "http://127.0.0.1:9380/ocr?layout=invoice" \
  -F "file=@path_to_image_or_pdf"
```

## Ví dụ response thực tế (invoice, rút gọn)

```json
{
  "text": "CHY CTY CP BB HUE TÀI DA LAT ...",
  "layout": "invoice",
  "pages": [
    {
      "page": 1,
      "text": "CHY CTY CP BB HUE TÀI DA LAT ...",
      "invoice": {
        "fields": {
          "phones": ["02633545088"],
          "tax_code": "3300854978",
          "total_amount": {
            "value": 234622,
            "raw": "234,622",
            "line": "TÔNG CỘNG VND 234,622"
          }
        },
        "sections": {
          "header": { "text": "CHY CTY CP BB HUE TÀI DA LAT PALLA ..." },
          "items": { "text": "1.000 BIC X 10,000 10,000 ..." },
          "totals": { "text": "TÔNG CỘNG VND 234,622" },
          "footer": { "text": "" }
        },
        "items": [
          {
            "name": "000 BIC X 10,000",
            "quantity": "1.000",
            "unit_price": 10000,
            "line_total": 10000
          }
        ]
      }
    }
  ],
  "invoice": { "...": "..." }
}
```

## BE C# cần xử lý như sau

- Thực hiện gọi API theo mục "Input nhận vào".
- BE chỉ cần lưu response JSON thô ở folder services của import và trả lại cho FE.
- Xử lý layout JSON (như 2 file bạn đưa) nên để Frontend (FE) là chính.
- BE có thể trả thông báo "đã thêm hóa đơn thành công" sau khi OCR trả về thành công.
