def main():
    from ultralytics import YOLO
    from PIL import Image
    
    # the trained model
    model = YOLO(r"runs\detect\train19\weights\best.pt")

    # testing
    results = model(r"..\css-data\test\cee_faculty.png")

    # Visualize the results
    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        _ = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

        # Save results
        r.save(filename=f'results{i}.jpg')
        

if __name__ == '__main__':
    main()