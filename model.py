import cv2
import os
from ultralytics import YOLO
from roboflow import Roboflow
from PIL import Image


class ModelDetect(YOLO):
    def __init__(self, model_path=None):
        super().__init__('yolov8n-seg.pt')
        self.path_data = ''
        if model_path != None:
            self.pre_model = YOLO(model_path)

    def Train(self, path_data, tr=True):
        self.path_data = path_data
        if tr:
            try:
                super().train(data=f"{self.path_data}/data.yaml", epochs=5, imgsz=640)
            except Exception as e:
                print("Error:", e)

    def PredictTest(self,data):
        self.data = data
        results = self.pre_model(self.data)
        result = results[0]
        if result:
            for box in result.boxes:

                class_id = result.names[box.cls[0].item()]
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                conf = round(box.conf[0].item(), 2)
                print("Object type:", class_id)
                print("Coordinates:", cords)
                print("Probability:", conf)
                print("---")

            image = Image.fromarray(result.plot()[:, :, ::-1])
            image.show()
            return class_id, cords, conf

    def Predict(self, data):
        self.data = data
        results = self.pre_model(self.data)
        detections = []
        if results:
            for box in results[0].boxes:
                class_id = results[0].names[box.cls[0].item()]
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                conf = round(box.conf[0].item(), 2)
                detections.append((class_id, cords, conf))
            image = Image.fromarray(result.plot()[:, :, ::-1])
            image.show()
        return detections
