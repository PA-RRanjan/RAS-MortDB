from ultralytics import YOLO
import argparse, os

def run_inference(image_path, model_path, conf=0.25):
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf, iou=0.45, save=True, save_conf=True, verbose=False)
    result = results[0]
    dead = sum(1 for c in result.boxes.cls if result.names[int(c)] == "Dead")
    live = sum(1 for c in result.boxes.cls if result.names[int(c)] == "Live")
    print(f"Dead: {dead}  Live: {live}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="weights/pytorch/yolo26n_best.pt")
    parser.add_argument("--conf", default=0.25, type=float)
    args = parser.parse_args()
    run_inference(args.image, args.model, args.conf)
