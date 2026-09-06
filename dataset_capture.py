"""
FortsBot — Dataset Capture Utility (standalone)
-----------------------------------------------
Мини-утилита для сбора датасета: ловит клик правой кнопкой мыши и сохраняет
скриншот текущего экрана (регион захвата) в dataset/зеленка/ как цветной PNG.

Зачем отдельно от бота: сбор кадров для обучения — отдельная задача, которой
не место в игровом цикле. Раньше этот код был вшит в state_reader/main.py, из-за
чего постоянно приходилось комментировать основные модули. Теперь это
самостоятельный скрипт, который не трогает OCR/мозг/ввод.

Запуск:
    python dataset_capture.py

Остановка:
    нажми 'q' (через Win32 GetAsyncKeyState — права админа не нужны).

Необязательные аргументы (регион захвата; по умолчанию — весь основной монитор):
    --top Y --left X --width W --height H
"""

import os
import time
import ctypes
import argparse

import cv2
import numpy as np
import mss

# Виртуальные коды клавиш Win32
VK_RBUTTON = 0x02   # правая кнопка мыши
VK_Q       = 0x51   # 'q' — выход

# Куда сохраняем кадры (подпапка внутри dataset/)
SAVE_DIR = os.path.join("dataset", "зеленка")


def _key_down(vk):
    """True, пока клавиша физически зажата (Win32 GetAsyncKeyState)."""
    return (ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000) != 0


def _save_png(filename, bgr_frame):
    """
    Пишет BGR-кадр в PNG через imencode + байтовую запись.

    Используем imencode вместо cv2.imwrite, потому что cv2.imwrite на Windows
    не умеет пути с кириллицей (папка "зеленка") — может молча не сохранить.
    imencode кодирует в память, а файл пишем обычным Python-ом (Unicode-путь).
    """
    ok, buf = cv2.imencode(".png", bgr_frame)
    if not ok:
        return False
    with open(filename, "wb") as f:
        f.write(buf.tobytes())
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Сохранение скриншотов по правому клику в dataset/зеленка/"
    )
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--left", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    with mss.MSS() as sct:
        monitor = sct.monitors[1]  # основной монитор: {'left','top','width','height'}
        if None not in (args.top, args.left, args.width, args.height):
            region = {"top": args.top, "left": args.left,
                      "width": args.width, "height": args.height}
        else:
            region = {"top": monitor["top"], "left": monitor["left"],
                      "width": monitor["width"], "height": monitor["height"]}

        print("=" * 60)
        print("    FORTSBOT DATASET CAPTURE (правый клик = скриншот)")
        print("=" * 60)
        print(f"Регион захвата: {region}")
        print(f"Сохраняю в:     {os.path.abspath(SAVE_DIR)}")
        print("Правый клик — сохранить кадр.")
        print("'q' — выход.\n")

        rbutton_down = False  # для детекта фронта (отпущено -> нажато)
        counter = 0

        try:
            while True:
                if _key_down(VK_Q):
                    print("\n[EXIT] Выход по нажатию клавиши.")
                    break

                right_down = _key_down(VK_RBUTTON)

                # Срабатываем только на самом нажатии (фронт), а не на удержании.
                if right_down and not rbutton_down:
                    shot = np.array(sct.grab(region))                     # BGRA
                    frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)        # -> BGR
                    counter += 1
                    filename = os.path.join(SAVE_DIR, f"frame_{int(time.time() * 1000)}.png")
                    if _save_png(filename, frame):
                        print(f"[{counter}] Сохранено: {filename}")
                    else:
                        print(f"[{counter}] Ошибка сохранения: {filename}")

                rbutton_down = right_down
                time.sleep(0.01)  # опрос ~100 Гц, не грузим CPU впустую

        except KeyboardInterrupt:
            print("\n[EXIT] Остановлено по Ctrl+C.")


if __name__ == "__main__":
    main()
