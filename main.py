from model import ModelDetect

def main():
    model_path = '/best.pt'
    image_path = 'D:/งานลูกค้า/Apiwat Taninkorn/datatraing/train/images/Ko-23-_jpg.rf.c220d9044dfd63ec3e539c2f377989b2.jpg'
    # Initialize the model with the specified path
    model = ModelDetect(model_path)
    # Run the prediction on the given image path
    model.PredictTest(image_path)
if __name__ == "__main__":
    main()