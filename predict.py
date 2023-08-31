from PIL import Image

from ultralytics import YOLO


def predict():
    # Load a model
    model = YOLO('runs/detect/train4/weights/best.pt')  # pretrained YOLOv8n model

    # Define path to the image file
    source = 'F:/BrowserDownloads/datasets/CTCyolov8/test/images/Model_system-Slide1-Mixture-1224-2021_r1c1x1y7_jpg' \
             '.rf.225ba595a034402c7033f412fa81d610.jpg'

    # Run batched inference on a list of images
    results = model.predict(source)  # return a list of Results objects
    print(results)
    # for r in results:
    #     im_array = r.plot()  # plot a BGR numpy array of predictions
    #     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    #     im.show()  # show image
    #     im.save('results.jpg')  # save image
    # Process results list
    # for result in results:
    #     boxes = result.boxes  # Boxes object for bbox outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs


if __name__ == '__main__':
    predict()