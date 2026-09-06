from pathlib import Path
import shutil
import cv2
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
CONF = 0.3  # порог уверенности

# Тестовые картинки (старые + новые).
TEST_DIR = HERE / "test_imaige"
images = sorted(TEST_DIR.glob("*.png"))

# Текущая лучшая модель.
MODEL_FILE = "best.pt"
OUT_DIR = "best_test"


def main():
    model = YOLO(HERE / MODEL_FILE)
    out = HERE / OUT_DIR
    if out.exists():
        shutil.rmtree(out)  # чистим прошлый прогон
    out.mkdir(exist_ok=True)

    print(f"МОДЕЛЬ: {MODEL_FILE}  ->  {OUT_DIR}/")

    total = 0
    total_conf = 0.0
    for img in images:
        r = model(img, conf=CONF)[0]
        confs = r.boxes.conf.tolist()
        n = len(confs)
        total += n
        total_conf += sum(confs)
        conf_str = ", ".join(f"{c:.2f}" for c in confs) if confs else "-"
        print(f"  {img.name}: объектов = {n}  (conf: {conf_str})")

        annotated = r.plot()
        cv2.imwrite(str(out / f"{img.stem}_detected.png"), annotated)

    avg_conf = total_conf / total if total else 0.0
    print(f"\nИТОГО: {total} детекций на {len(images)} картинках, средний conf = {avg_conf:.2f}")
    print(f"Результаты: {out}")


if __name__ == "__main__":
    main()
