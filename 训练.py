from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11m.pt")

    model.train(
        data=r"valorant.yaml",
        epochs=100,
        patience=30,
        imgsz=640,
        batch=-1,
        cache="ram",
        workers=1,

        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        perspective=0.0005,
        flipud=0.01,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        fraction=1.0,
        seed=42,
        device='cpu',  # ✅ 关键修改：GitHub Actions 无GPU，强制用CPU
        pretrained=True,
        verbose=True,

        save=True,
        save_period=-1,
        val=True,
        plots=True,
        name="valorant2.0",
        exist_ok=True
    )
