from pathlib import Path
import cv2
from ultralytics import YOLO

# Папка, в которой лежит этот скрипт
HERE = Path(__file__).resolve().parent

# 1. Загружаем обученную модель
model = YOLO(HERE / "best.pt")

# 2. Берём все картинки для проверки (любой .png, кроме результата детекции)
images = sorted(p for p in HERE.glob("*.png") if not p.stem.endswith("_detected"))
if not images:
    raise SystemExit("Нет картинок для проверки — положи .png в папку yolo_test/")

CONF = 0.3  # порог уверенности

for IMAGE in images:
    print(f"\n{'='*60}")
    print(f"Проверяю: {IMAGE.name}")
    print(f"{'='*60}")

    # 3. Инференс с заданным порогом уверенности
    results = model(IMAGE, conf=CONF)
    result = results[0]

    print(f"Найдено объектов: {len(result.boxes)}")
    for box in result.boxes:
        xyxy = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names.get(cls, str(cls))
        print(f"  {name} (conf={conf:.3f}) — xyxy: {xyxy}")

    # 4. Рисуем выделения и сохраняем рядом с исходником
    annotated = result.plot()
    out_path = HERE / f"{IMAGE.stem}_detected.png"
    cv2.imwrite(str(out_path), annotated)
    print(f"Сохранено: {out_path.name}")
