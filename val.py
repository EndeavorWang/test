from ultralytics import YOLO


def val():
    # Load a model
    model = YOLO('ctc/yolov8n_1000_t925_v100/weights/best.pt')  # load a custom model

    # Validate the model
    metrics = model.val(data='F:/BrowserDownloads/datasets/CTC.v4i.yolov8/data.yaml')  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category

if __name__ == '__main__':
    val()