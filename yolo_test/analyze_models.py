import pathlib
import logging
import torch
import numpy as np

logging.getLogger('ultralytics').setLevel(logging.ERROR)

HERE = pathlib.Path(__file__).resolve().parent
MODEL = HERE / 'best.pt'


def main():
    print(f'Анализирую модель: {MODEL}')
    ckpt = torch.load(MODEL, map_location='cpu', weights_only=False)
    ta = dict(ckpt.get('train_args', {}))

    print('\n' + '=' * 70)
    print('1) МЕТАДАННЫЕ ТРЕНИРОВКИ (train_args)')
    print('=' * 70)
    fields = ['name', 'data', 'model', 'imgsz', 'batch', 'epochs', 'patience',
              'lr0', 'lrf', 'optimizer', 'pretrained', 'resume', 'mosaic',
              'degrees', 'scale', 'fliplr', 'hsv_h', 'hsv_s', 'hsv_v', 'amp',
              'fraction', 'seed', 'single_cls']
    for f in fields:
        print(f'  {f:12s}: {ta.get(f)}')
    print('  epoch (последняя эпоха):', ckpt.get('epoch'))
    print('  date (дата сохранения):', ckpt.get('date'))

    print('\n' + '=' * 70)
    print('2) АРХИТЕКТУРА МОДЕЛИ')
    print('=' * 70)
    m = ckpt.get('model') or ckpt.get('ema')
    if m is not None:
        nparams = sum(p.numel() for p in m.parameters())
        names = getattr(m, 'names', None)
        nlayers = len(m.model) if hasattr(m, 'model') else '?'
        print(f'  параметров = {nparams:,}, слоёв = {nlayers}, классы = {names}')

    print('\n' + '=' * 70)
    print('3) УВЕРЕННОСТЬ ДЕТЕКЦИЙ НА ТЕСТОВЫХ ФОТО')
    print('=' * 70)
    from ultralytics import YOLO
    imgs = sorted((HERE / 'test_imaige').glob('*.png'))
    model = YOLO(MODEL)
    confs = []
    for img in imgs:
        r = model(img, conf=0.3, verbose=False)[0]
        confs += r.boxes.conf.tolist()
    confs = np.array(confs)
    print(f'  детекций = {len(confs)}, mean = {confs.mean():.3f}, '
          f'median = {np.median(confs):.3f}, min = {confs.min():.3f}, max = {confs.max():.3f}')
    for thr in (0.3, 0.5, 0.7):
        print(f'    conf >= {thr}: {(confs >= thr).sum()}')


if __name__ == '__main__':
    main()
