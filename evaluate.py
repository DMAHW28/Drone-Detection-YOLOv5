import torch
import argparse
from ultralytics import YOLO

device_intel = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_apple_silicon = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "models/drone_yolov5s2/weights/best.pt"
model = YOLO(MODEL_PATH)

def detection(args):
    image_path = str(args.image_path)
    device = device_apple_silicon if args.device == "apple" else device_intel
    prediction = model.predict(source=image_path, conf=0.5, save=False, device=device)
    for pred in prediction:
        pred.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone detection")
    parser.add_argument("--image_path", type=str, required=True, help="Image name")
    parser.add_argument("--device", type=str, default="apple", help="Device type")
    detection(parser.parse_args())