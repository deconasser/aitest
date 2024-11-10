from mmdet.apis import init_detector, inference_detector
import os
import numpy as np

# Đường dẫn đến file config và checkpoint của CO-DETR
config_file = "./helios-cfg/co_dino_5scale_swin_large_16e_o365tococo.py"
checkpoint_file = "./helios/epoch_1.pth"
input_folder = "./public test"

# Kích thước ảnh thực tế (cần thay đổi theo kích thước ảnh của bạn)
image_width = 1280
image_height = 720

# Ngưỡng confidence để lọc
confidence_threshold = 0.4

# Tạo file mới trong thư mục /kaggle/working
output_file_path = "./predict.txt"

# Khởi tạo mô hình
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Kiểm tra nếu file tồn tại trước đó, xóa đi để tạo file mới
if os.path.exists(output_file_path):
    os.remove(output_file_path)

# Mở file để ghi kết quả và kiểm tra lỗi
try:
    with open(output_file_path, 'w') as f:
        # Lặp qua từng ảnh trong thư mục
        for image_file in os.listdir(input_folder):
            if image_file.endswith('.jpg'):  # Kiểm tra định dạng ảnh
                image_path = os.path.join(input_folder, image_file)

                # Chạy inference
                result = inference_detector(model, image_path)

                # Kiểm tra nếu không có kết quả dự đoán
                if result is None or len(result) == 0:
                    print(f"Không có kết quả dự đoán cho ảnh: {image_file}")
                    continue

                # Lấy tên ảnh để ghi vào file
                image_name = image_file

                # Xử lý kết quả đầu ra: mỗi lớp có thể có nhiều bbox
                for class_id, bboxes in enumerate(result):
                    for bbox in bboxes:
                        x1, y1, x2, y2, confidence_score = bbox

                        # Chỉ giữ lại các bbox có confidence score lớn hơn ngưỡng
                        if confidence_score < confidence_threshold:
                            continue

                        # Tính toán các giá trị trung tâm, chiều rộng và chiều cao của bbox
                        x_center = (x1 + x2) / 2.0
                        y_center = (y1 + y2) / 2.0
                        width = x2 - x1
                        height = y2 - y1

                        # Chuẩn hóa tọa độ x_center, y_center, width, height
                        x_center /= image_width
                        y_center /= image_height
                        width /= image_width
                        height /= image_height

                        # Định dạng kết quả với 5 chữ số sau dấu chấm
                        line = f"{image_name} {class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence_score:.6f}\n"
                        f.write(line)

except Exception as e:
    print(f"Lỗi khi ghi vào file: {e}")

print(f"Kết quả đã được ghi vào file: {output_file_path}")