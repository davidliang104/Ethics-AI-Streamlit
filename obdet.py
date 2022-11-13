import subprocess
import torch

# subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5'])  # Clone YOLOv5 repo
# %cd /content/yolov5
# !pip install -qr requirements.txt  # Install dependencies

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")


subprocess.run(['python', 'yolov5/detect.py', '--weights', 'best_iaug720.pt', '--img', '720', '--conf', '0.4', '--source', 'images', '--save-txt', '--save-conf', '--exist-ok'])

