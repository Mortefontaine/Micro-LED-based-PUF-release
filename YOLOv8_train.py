from ultralytics import YOLO

def train():
    dataset_yaml = r'D:/Research/PUF_micro-LED/videos3/frames_YOLO/dataset.yaml'

    yaml_text = """
train: D:/Research/PUF_micro-LED/videos3/frames_YOLO/images/train
val: D:/Research/PUF_micro-LED/videos3/frames_YOLO/images/val
nc: 1
names: ['microled']
    """

    with open(dataset_yaml, 'w') as f:
        f.write(yaml_text.strip())

    print(f"✅ dataset.yaml created: {dataset_yaml}")

    model = YOLO('yolov8n.pt')

    model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        workers=2  # Start with 2 workers; adjust based on your machine
    )

    print('✅ Training complete! Best model saved to runs/detect/train/weights/best.pt')

if __name__ == '__main__':
    train()
