from ultralytics import YOLO
def main():
    model = YOLO("yolov8n.pt")

    # USA MODELO
    model.train(data="construction.yaml", epochs=30, device=0)
    metrics = model.val()

if __name__ == '__main__':
    main()
