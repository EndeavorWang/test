from ultralytics import YOLO
import wandb




def train():
    wandb.init(project="my-testproject")
    model = YOLO('yolov8n.pt')
    results = model.train(data='F:/BrowserDownloads/datasets/CTC-CAF/data.yaml', batch=8, epochs=3, imgsz=640,
                          project='ctc-caf', name='test')


if __name__ == '__main__':
    train()