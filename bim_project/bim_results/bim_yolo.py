def main():
    from ultralytics import YOLO
    
    # do not create a new model and train from scratch
    # model = YOLO('yolov8n.yaml')

    # Load pretrained YOLO model (smallest one: yolov8n.pt)
    model = YOLO('yolov8n.pt')

    # Train the model with 'ppe_data'; 120 epochs
    _ = model.train(data="ppe_data.yaml", epochs=120)

    # Save the model
    _ = model.export(format='onnx')

if __name__ == '__main__':
    main()