import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image

# 1. Cấu hình mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2 

model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
model.load_state_dict(torch.load("dragon_fruit_path_coco.pth")) 
model.to(device)
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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(input_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    output_predictions = output.argmax(0).byte().cpu().numpy()
    mask = cv2.resize(output_predictions, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
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
        cx = int(np.mean(indices))
        
        # Tính toán điều khiển
        center_x = W // 2
        error = cx - center_x
        MAX_ANGLE = 30 
        KP = 0.15      
        steering_angle = max(min(error * KP, MAX_ANGLE), -MAX_ANGLE)
        speed = 100 if abs(steering_angle) < 10 else 60
        
        # --- VẼ MŨI TÊN CHỈ HƯỚNG ---
        # Điểm bắt đầu: Giữa cạnh dưới ảnh (vị trí xe)
        start_point = (W // 2, H - 20)
        # Điểm kết thúc: Tâm đường tại vạch look_ahead
        end_point = (cx, look_ahead_line)
        
        # Vẽ mũi tên (Màu vàng, độ dày 5, đầu mũi tên lớn 0.3)
        cv2.arrowedLine(combined_frame, start_point, end_point, (0, 255, 255), 5, tipLength=0.1)
        
        # Vẽ các thông tin bổ trợ
        cv2.line(combined_frame, (0, look_ahead_line), (W, look_ahead_line), (255, 255, 0), 1)
        cv2.circle(combined_frame, (cx, look_ahead_line), 8, (0, 0, 255), -1)
        
        cv2.putText(combined_frame, f"Steer: {int(steering_angle)} deg", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(combined_frame, f"Speed: {speed}", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.putText(combined_frame, "PATH LOST!", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Hiển thị kết quả
    cv2.imshow("Robot Vision", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()