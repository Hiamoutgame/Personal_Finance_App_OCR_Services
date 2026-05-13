# Sử dụng một base image Python
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép file requirements.txt và cài đặt các dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn và các tài nguyên cần thiết vào container
COPY . .

# Mở port mà ứng dụng sẽ chạy
EXPOSE 6969

# Lệnh để chạy ứng dụng khi container khởi động
# Chạy trên host 0.0.0.0 để có thể truy cập từ bên ngoài container
# Render.com sẽ tự động cung cấp biến môi trường PORT
CMD ["uvicorn", "ocr_api:app", "--host", "0.0.0.0", "--port", "6969"]
