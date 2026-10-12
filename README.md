# OCR Service

Service FastAPI đọc chữ tiếng Việt từ ảnh/PDF, dùng DeepDoc và VietOCR ONNX trên CPU.

## Cấu trúc

```text
ocr_api.py                  Entry point và ASGI app của service
ocr_cli.py                  Công cụ chạy thử (ocr, recognizer, pipeline)
pyproject.toml              Package và toàn bộ dependencies
src/ocr_service/
  application.py            FastAPI factory và lifespan
  api/                      Routes, schemas
  services/                 Ảnh/PDF, OCR pipeline, quy tắc hóa đơn
  engines/                  ONNX, layout, bảng, PyTorch cho CLI
  core/                     Settings, paths, logging, ONNX sessions
  cli/                      Logic của các lệnh CLI
  vendor/vietocr/           Source VietOCR
models/onnx/                Models DeepDoc
models/vietocr/             Config, vocabulary và trọng số
img/, output/               Ảnh mẫu và kết quả mẫu
tests/                      Kiểm thử API, CLI, đối chiếu bản gốc
```

## Cài đặt và chạy local

Dùng Python 3.12; Docker mặc định 3.12.10. Chạy từ thư mục dự án:

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e .
./.venv/Scripts/python.exe ocr_api.py
```

Linux/macOS: thay đường dẫn Python bằng `.venv/bin/python`.
Swagger: http://localhost:6969/docs. Health: http://localhost:6969/health.

Dependencies khai báo tại một nơi duy nhất là `pyproject.toml`: service ở `dependencies`, phần thêm ở extras `cli` và `dev`.

```powershell
./.venv/Scripts/python.exe ocr_api.py --host 127.0.0.1 --port 8080
./.venv/Scripts/python.exe -m uvicorn ocr_api:app --host 0.0.0.0 --port 6969
```

Host/port của entrypoint: CLI > ENV `HOST`/`PORT` > `0.0.0.0:6969`.
Khi gọi Uvicorn trực tiếp, dùng cờ của Uvicorn để chọn host/port.
Package đã cài thì có thể chạy entrypoint bằng đường dẫn tuyệt đối từ thư mục khác.
`.env.example` chỉ mô tả biến môi trường; service không tự đọc file `.env`.

## API

`POST /ocr?layout=none|invoice|document`: multipart/form-data, trường bắt buộc `file`. Mỗi request nhận một file.
Nhận PDF và ảnh Pillow đọc được như PNG, JPG, WEBP.

| Chế độ | Response |
|---|---|
| none (mặc định) | Chỉ `text` |
| invoice | `text`, `layout`, `pages[].invoice`; shortcut `invoice` khi một trang |
| document | `text`, `layout`, `pages[].layout_regions`; shortcut `layout_regions` khi một trang |

PDF nhiều trang ghép text bằng một dòng trống, trang đánh số từ 1, không có shortcut cấp gốc.
Invoice có `fields`, `sections`, `items`, `boxes`, `image`, `layout_config`, `text`.
Tổng tiền nếu tìm được nằm ở `invoice.fields.total_amount.value`; field có thể thiếu.
Bounding box: `[x0,y0,x1,y1]`, đơn vị pixel của ảnh; PDF render 216 DPI.
Quy tắc hóa đơn và điểm OCR `score=1` được giữ từ bản cũ; điểm này không phải độ tin cậy thực của model.
File rỗng, ảnh không đọc được hoặc PDF hỏng trả 400; thiếu file hoặc layout không hợp lệ trả 422.
`GET /health` trả `{"status":"ok"}` và không chạy OCR.

```powershell
curl.exe -F "file=@img/hoadon2.png" "http://localhost:6969/ocr?layout=invoice"
```

Service xử lý trong bộ nhớ, không lưu upload/kết quả, không cần database/Redis/storage/LLM.
OCR nạp lúc startup; layout nạp khi cần. Một process xử lý một job OCR tại một thời điểm, PDF từng trang.
Model nằm trong `models/`; không có tải model qua mạng khi chạy. Thiếu model OCR thì startup thất bại.
Dependencies service được cố định theo bộ phiên bản đã đối chiếu kết quả; nâng phiên bản cần chạy lại integration test.

## Kết nối backend C#

Backend gọi service qua HttpClient tại `<Ocr:BaseUrl>/ocr?layout=...`, gửi multipart tên `file`.

```json
{
  "Ocr": {
    "BaseUrl": "http://localhost:6969",
    "Layout": "invoice",
    "TimeoutSeconds": 300
  }
}
```

Base URL không bao gồm `/ocr`. Local: `http://localhost:6969`; backend trong cùng Compose: `http://server:6969`; Render: URL HTTPS của service.
Chọn `Ocr:Layout=invoice` để nhận boxes và thông tin hóa đơn.
Timeout 300 giây là điểm bắt đầu cho thử nghiệm Render Free; điều chỉnh theo cold start và thời gian OCR thực tế.

```csharp
using var form = new MultipartFormDataContent();
using var content = new StreamContent(fileStream);
content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/png");
form.Add(content, "file", fileName);
using var response = await httpClient.PostAsync(
    $"{baseUrl.TrimEnd('/')}/ocr?layout=invoice", form, cancellationToken);
var rawJson = await response.Content.ReadAsStringAsync(cancellationToken);
response.EnsureSuccessStatusCode();
```

PDF dùng content type `application/pdf`.
Backend xử lý non-success status, timeout và mất kết nối; HTTP 200 chỉ xác nhận OCR thành công, không xác nhận đã tạo giao dịch tài chính.
Service không lưu file/JSON hoặc ghi database. Backend quyết định lưu kết quả và bước review/confirm.

## Docker

```sh
docker compose up --build
```

Swagger: http://localhost:6969/docs. Tên service: `server`. Port mapping: `6969:6969`.
Image chạy `python ocr_api.py` bằng user không phải root, chỉ cài dependencies của service và chỉ chứa các model OCR/layout mà API cần.
Nếu checkout chứa Git LFS pointer, chạy `git lfs pull` trước khi build. Startup báo lỗi rõ ràng với file pointer thay vì tự tải model.
Healthcheck gọi `/health`; log ra stdout/stderr. Chỉ có một Uvicorn worker. Không cần database hay persistent volume.

### Đo bộ nhớ

```sh
docker build -t ocr-service .
docker run -d --name ocr-memory-test --memory=512m --memory-swap=512m -p 6970:6969 ocr-service
docker logs ocr-memory-test
curl -F 'file=@img/hoadon2.png' 'http://localhost:6970/ocr?layout=none'
curl -F 'file=@img/hoadon2.png' 'http://localhost:6970/ocr?layout=invoice'
curl -F 'file=@img/hoadon2.png' 'http://localhost:6970/ocr?layout=document'
docker stats --no-stream ocr-memory-test
docker inspect --format '{{.State.OOMKilled}}' ocr-memory-test
docker rm -f ocr-memory-test
```

Windows: dùng `curl.exe`. Nên thử thêm PDF nhiều trang thật và request đồng thời; ghi lại thời gian startup, thời gian request và đỉnh bộ nhớ.
Một lần đọc `docker stats --no-stream` không phải là đỉnh bộ nhớ: lấy mẫu trong lúc request chạy hoặc đọc cgroup `memory.peak` nếu có.
Không kết luận phù hợp Render Free chỉ từ kích thước file model hay số đo trên Windows.

Báo cáo bộ nhớ process có thể lặp lại (giá trị đỉnh là cộng dồn):

```sh
python tests/benchmark_runtime.py --input img/hoadon2.png
```

Để chạy báo cáo đó trong image, bind-mount script benchmark và file đầu vào, truyền `--project-root /app` và áp cùng giới hạn 512 MB. Model vẫn nằm trong image. RSS của process khác với tổng bộ nhớ container; cần kiểm tra thêm trạng thái cgroup/OOM.

```sh
docker run --rm --memory=512m --memory-swap=512m -v "$PWD/tests/benchmark_runtime.py:/app/benchmark_runtime.py:ro" -v "$PWD/img/hoadon2.png:/app/input.png:ro" ocr-service python /app/benchmark_runtime.py --project-root /app --input /app/input.png
```

Số đo tham khảo trên Windows (12/10/2026, `img/hoadon2.png`, không nạp PyTorch): startup 285 MiB; đỉnh cộng dồn `none` 327 MiB, `invoice` 329 MiB, `document` 531 MiB. Chế độ `document` vượt 512 MiB trong lần đo này; phép đo trong container Linux giới hạn 512 MB chưa được thực hiện.

## Render

1. Push source và file model thật (không phải LFS pointer) lên Git repository.
2. Tạo Blueprint từ `render.yaml`. File này định nghĩa web service với `runtime: docker`, `plan: free` và `healthCheckPath: /health`.
3. Nếu thư mục này nằm trong repository lớn hơn, đặt Render root directory là thư mục OCR service; giữ Dockerfile và context tương đối theo root đó.
4. Render cấp `PORT`; entrypoint lắng nghe `0.0.0.0:$PORT`. Không hardcode port Render trong Blueprint.
5. Kiểm tra log startup, `/health`, `/docs` và cả ba chế độ OCR trên URL HTTPS của service.
6. Đặt `Ocr:BaseUrl` của backend thành URL đó, không kèm `/ocr`.

Render Free có 512 MB RAM, ngủ sau 15 phút không có request và không giữ thay đổi filesystem lúc chạy. Service đóng gói model vào image và không lưu upload.
Nếu số đo cho thấy thiếu bộ nhớ, ghi lại workload lỗi và lượng bộ nhớ cần trước khi chọn gói compute khác.

Tham khảo: [Docker deployment](https://render.com/docs/docker), [port binding](https://render.com/docs/web-services), [Blueprint fields](https://render.com/docs/blueprint-spec), [Free limitations](https://render.com/docs/free).

## CLI

Công cụ chạy thử, không cần cho service. Cài thêm dependencies PyTorch cho lệnh `ocr`, `recognizer --mode tsr` và `pipeline`:

```powershell
./.venv/Scripts/python.exe -m pip install -e ".[cli]"
./.venv/Scripts/python.exe ocr_cli.py ocr --inputs img/hoadon2.png --output_dir output --layout invoice
./.venv/Scripts/python.exe ocr_cli.py recognizer --inputs img/hoadon2.png --output_dir output --mode layout --threshold 0.5
./.venv/Scripts/python.exe ocr_cli.py recognizer --inputs img/hoadon2.png --output_dir output --mode tsr --threshold 0.5
./.venv/Scripts/python.exe ocr_cli.py pipeline --inputs img/hoadon2.png --output_dir output --threshold 0.5
```

`--inputs` nhận ảnh, PDF hoặc thư mục. Lệnh `ocr` mặc định `invoice`, xuất ảnh, `.txt`, `.text.json`, `.json`.
`recognizer --mode layout` xuất ảnh; `--mode tsr` thêm `.md`; `pipeline` xuất `_full.md`.
CLI giữ backend PyTorch CPU; API dùng ONNX CPU. Riêng `recognizer --mode layout` không cần PyTorch.
Log hiện ở console và `log/<tên-lệnh>.log`.
Ngưỡng phát hiện YOLOv10 giữ nguyên 0.08; ngưỡng CLI còn dùng khi vẽ/chọn vùng. Thuật toán pipeline được giữ, gồm giới hạn xử lý mask hiện có.

## Kiểm thử

```powershell
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m pytest -q -m "not integration"
```

Integration test dùng model thật và fixture trước refactor để so text, boxes, invoice JSON, layout.
Unit test dùng engine giả để kiểm tra contract, file lỗi, nhiều PDF, request đồng thời, health và output CLI.

Nguồn model/source: [DeepDoc](https://github.com/infiniflow/ragflow/tree/main/deepdoc), [VietOCR](https://github.com/pbcquoc/vietocr).
