import torch
import cv2
import numpy as np
import os
from torchvision import models, transforms
from PIL import Image

# 1. Cấu hình mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2 

model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))

# Load best model thay vì epoch cuối
model_path = "dragon_fruit_path_coco_best.pth"
if not os.path.exists(model_path):
    print(f"Cảnh báo: Không tìm thấy {model_path}. Thử load dragon_fruit_path_coco.pth...")
    model_path = "dragon_fruit_path_coco.pth"

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Đã load model: {model_path}")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file model. Vui lòng chạy train.py trước.")
    exit(1)

model.to(device)

# Sử dụng FP16 (Half precision) nếu có CUDA để tăng tốc độ Inference
if device.type == "cuda":
    model.half()

model.eval()

# 2. Cấu hình Transform
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Mở Video
video_path = "IMG_0974.MOV"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Lỗi: Không thể mở video {video_path}")
    exit(1)

# Các tham số cho thuật toán điều khiển (PD Controller)
MAX_ANGLE = 30 
KP = 0.15
KD = 0.05 
previous_error = 0

# Tham số cho Exponential Moving Average (EMA) Filter
smoothed_steering_angle = 0
ALPHA = 0.7 # Trọng số cho giá trị cũ, càng cao càng mượt nhưng phản hồi chậm

# Tham số cho tính năng Quay đầu (U-Turn)
is_turning = False
frames_path_lost = 0
LOST_THRESHOLD = 15      # Số frame liên tiếp không thấy đường thì bắt đầu rẽ
TURN_SPEED = 50          # Tốc độ khi đang rẽ chữ U
TURN_DIRECTION = 'RIGHT' # Hướng rẽ ưu tiên ('RIGHT' hoặc 'LEFT')
# Cài đặt góc rẽ: rẽ phải là dương, rẽ trái là âm (hoặc tuỳ thuộc vào quy ước của robot)
TURN_ANGLE = MAX_ANGLE if TURN_DIRECTION == 'RIGHT' else -MAX_ANGLE

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(input_image).unsqueeze(0).to(device)
    
    if device.type == "cuda":
        input_tensor = input_tensor.half() # Ép kiểu input tensor về FP16
    
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    output_predictions = output.argmax(0).byte().cpu().numpy()
    mask = cv2.resize(output_predictions, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # --- POST-PROCESSING MASK ---
    # 1. Morphological Operations (Opening -> Closing) để xóa nhiễu và nối các đốm vỡ
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 2. Tìm Contour lớn nhất để vứt bỏ các vệt nhiễu nằm rời rạc ngoài đường đi
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 1, thickness=cv2.FILLED)
        mask = clean_mask
        
    # --- XỬ LÝ HIỂN THỊ ---
    green_overlay = np.zeros_like(frame)
    green_overlay[mask == 1] = [0, 255, 0] 
    combined_frame = cv2.addWeighted(frame, 1.0, green_overlay, 0.4, 0)
    
    W = frame.shape[1]
    H = frame.shape[0]

    look_ahead_line = int(H * 0.6) 
    path_at_line = mask[look_ahead_line, :]
    indices = np.where(path_at_line == 1)[0]

    if len(indices) > 0:
        # Reset biến đếm nếu tìm thấy đường
        frames_path_lost = 0
        if is_turning:
            is_turning = False # Thoát khỏi trạng thái quay đầu

        cx = int(np.mean(indices))
        
        # --- TÍNH TOÁN ĐIỀU KHIỂN (PD Controller) ---
        center_x = W // 2
        error = cx - center_x
        
        derivative = error - previous_error
        raw_steering_angle = (error * KP) + (derivative * KD)
        raw_steering_angle = max(min(raw_steering_angle, MAX_ANGLE), -MAX_ANGLE)
        
        previous_error = error
        
        # --- LÀM MƯỢT GÓC LÁI (EMA Filter) ---
        smoothed_steering_angle = (ALPHA * smoothed_steering_angle) + ((1 - ALPHA) * raw_steering_angle)
        
        speed = 100 if abs(smoothed_steering_angle) < 10 else 60
        
        # --- VẼ MŨI TÊN CHỈ HƯỚNG ---
        start_point = (W // 2, H - 20)
        end_point = (cx, look_ahead_line)
        
        cv2.arrowedLine(combined_frame, start_point, end_point, (0, 255, 255), 5, tipLength=0.1)
        
        # Vẽ các thông tin bổ trợ
        cv2.line(combined_frame, (0, look_ahead_line), (W, look_ahead_line), (255, 255, 0), 1)
        cv2.circle(combined_frame, (cx, look_ahead_line), 8, (0, 0, 255), -1)
        
        # Hiển thị text lên màn hình
        cv2.putText(combined_frame, f"Steer (Raw): {int(raw_steering_angle)} deg", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(combined_frame, f"Steer (Smooth): {int(smoothed_steering_angle)} deg", (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(combined_frame, f"Speed: {speed}", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        frames_path_lost += 1
        
        if frames_path_lost >= LOST_THRESHOLD:
            is_turning = True
            
        if is_turning:
            # Đang trong trạng thái U-Turn
            smoothed_steering_angle = TURN_ANGLE
            speed = TURN_SPEED
            
            # --- VẼ MŨI TÊN MINH HOẠ U-TURN ---
            start_point = (W // 2, H - 20)
            end_point = (W - 50 if TURN_DIRECTION == 'RIGHT' else 50, H - 20 - int(abs(TURN_ANGLE) * 2))
            cv2.arrowedLine(combined_frame, start_point, end_point, (0, 0, 255), 8, tipLength=0.2)
            
            cv2.putText(combined_frame, f"U-TURN TO NEXT ROW ({TURN_DIRECTION})", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.putText(combined_frame, f"Steer (Fixed): {int(smoothed_steering_angle)} deg", (20, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(combined_frame, f"Speed: {speed}", (20, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Chờ xác nhận mất đường (Debounce)
            cv2.putText(combined_frame, f"PATH LOST? (Wait {LOST_THRESHOLD - frames_path_lost} frames)", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 3)

    # Hiển thị kết quả
    cv2.imshow("Robot Vision", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()