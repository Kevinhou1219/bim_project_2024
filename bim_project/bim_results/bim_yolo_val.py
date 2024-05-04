def main():
    from ultralytics import YOLO
    
    # the trained model
    model = YOLO(r"runs\detect\train19\weights\best.pt")

    # run the model on validation set
    _ = model.val(data="ppe_data.yaml")

if __name__ == '__main__':
    main()