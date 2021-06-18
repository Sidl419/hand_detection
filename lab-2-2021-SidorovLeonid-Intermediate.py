import numpy as np
import cv2
import matplotlib.pyplot as plt
import os.path
from PIL import Image
from skimage.morphology import area_closing, skeletonize
import pandas as pd
from skan import Skeleton, summarize
from scipy.spatial.distance import pdist
from glob import glob
import time

def get_mask(filename):
    """
    На основе картинки получаем бинарную маску ладони

    filename - имя изображения в рабочей директории
    """
    img = cv2.imread(filename)

    # Преобразуем BGR в HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Применяем бинаризацию по порогу, чтобы выделить цвет ладони
    mask = cv2.inRange(hsv, np.array([0,0,60]), np.array([250,255,255]))

    # Применяем пространственные преобразования, чтобы избавиться от шума
    mask = area_closing(mask, 3000)

    # Применяем медианный фильтр, чтобы сгладить изображение
    mask = cv2.medianBlur(mask, 17)

    return mask

def get_endpoints(mask, min_bone, max_bone):
    """
    На основе бинарной маски строим скелет изображения и возвращаем 
    координаты его начальных и конечных вершин

    mask - бинарная маска (вывод функции get_mask)
    min_bone - минимально допустимая длина ребра в скелете изображения
    max_bone - максимально допустимая длина ребра в скелете изображения
    """
    # Проводим скелетизацию изображения
    # Таблица `branch_data` содержит всю необходимую информацию о скелете
    skeleton = skeletonize(mask // 255)
    branch_data = summarize(Skeleton(skeleton))
    
    # branch-type == 1 соответствует наружным рёбрам скелета
    temp = branch_data[(branch_data['branch-type'] == 1)]

    # Оставляем только рёбра нужной длины
    temp = temp[(temp['branch-distance'] > min_bone) & (temp['branch-distance'] < max_bone)]

    # Исключаем из рассмотрения вершины, находящиеся на границе изображения
    temp = temp[(temp['coord-dst-0'] > 10) & (temp['coord-dst-1'] > 30)]
    temp = temp[(temp['coord-dst-0'] < mask.shape[0] - 10) & (temp['coord-dst-1'] < mask.shape[1] - 10)]

    # Возвращаем два множества: внутренние и внешние вершины скелета
    dst = temp[['coord-dst-0', 'coord-dst-1']].reset_index(drop=True)
    src = temp[['coord-src-0', 'coord-src-1']][temp['skeleton-id'] == temp['skeleton-id'].mode()[0]].reset_index(drop=True)

    return dst, src

def get_points(mask):
    """
    На основе бинарной маски строим "линию пальцев" из 
    постановки задачи

    mask - бинарная маска (вывод функции get_mask)
    """
    # На основе бинарной маски строим два скелета:
    # скелет ладони и скелет окружающего её пространства
    dst_out, src_out = get_endpoints(mask, 110, 320)
    dst_in, src_in = get_endpoints(255 - mask, 70, 10000)

    # Если ладонь перевёрнута, то нам необходимо 
    # поменять местами множества внутренних и внешних 
    # вершин (особенности алгоритма скелетизации)
    if pdist(src_out.to_numpy()).sum() < 1000:
        t = src_out
        src_out = dst_out
        dst_out = t

        if pdist(dst_in.to_numpy()).sum() > 1200:
            t = src_in
            src_in = dst_in
            dst_in = t

    src_ids = src_out.to_numpy()
    dst_ids = dst_in.to_numpy()

    # Убираем из рассмотрения выбросы - точки, лежащие 
    # слишком далеко от основного скопления
    dist = src_out.apply(lambda x: np.power(src_ids - x.to_numpy(), 2).sum(), axis=1)
    while dist.max() > 8 * 10e4:
        src_out = src_out.drop([dist.idxmax()])
        dist = src_out.apply(lambda x: np.power(src_ids - x.to_numpy(), 2).sum(), axis=1)

    # Начальной точкой ломаной является конец большого пальца
    # Эта точка является выбросом по отношению к остальным четырём вершинам
    inter_dist = src_out.apply(lambda x: np.power(dst_ids - x.to_numpy(), 2).sum(), axis=1)
    outer_dist = src_out.apply(lambda x: np.power(dst_out.to_numpy() - x.to_numpy(), 2).sum(), axis=1)
    big = src_out.iloc[dist.idxmax()].to_numpy()
    src_out = src_out.drop([dist.idxmax()])

    # Однако мы можем ошибиться и выбрать точку на другом конце 
    # ребра большого пальца
    other = dst_out.iloc[dist.idxmax()].to_numpy()
    src_ids = np.delete(src_ids, dist.idxmax() - 1, 0)
    other_dist = np.power(src_ids - dst_out.iloc[dist.idxmax()].to_numpy(), 2).sum()
    inter_other_dist = np.power(dst_ids - dst_out.iloc[dist.idxmax()].to_numpy(), 2).sum()
    outer_other_dist = np.power(dst_out.to_numpy() - dst_out.iloc[dist.idxmax()].to_numpy(), 2).sum()
    dst_out = dst_out.drop([dist.idxmax()])

    # Если это так, то такая точка будет делить свою позицию с другой вершиной 
    # или находиться слишком близко к скоплению внутренних вершин рёбер
    any_dots = (dst_ids == big).sum()

    if (other_dist > dist.max()) and (outer_other_dist > outer_dist.iloc[dist.idxmax()]) or (inter_other_dist > inter_dist.iloc[dist.idxmax()]) or any_dots:
        res = [other]
    else:
        res = [big]

    dst_in = dst_in.to_numpy()
    src_out = src_out.to_numpy()
    dst_out = dst_out.to_numpy()

    # Насколько далеко назад мы заглядываем при поиске 
    # следующей точки ломаной
    look_back = np.full(min(len(dst_in), len(src_out), 4), 2)
    look_back[0] = 1

    # Если место наружней вершины чем-то занято, то 
    # на самом деле это внутренняя вершина, и нам надо 
    # заменить её на парную
    for i, el in enumerate(src_out):
        if (dst_out == el).prod(axis=1).sum():
            src_out[i] = dst_out[i]

    # Последовательный поиск следующих точек ломаной
    for i in look_back:
        # Ищем ближайшую точку основания пальца
        dst_ind = np.power(dst_in - res[-i], 2).sum(axis=1).argmin()
        res.append(dst_in[dst_ind])
        dst_in = np.delete(dst_in, dst_ind, 0)

        # Ищем ближайшую точку кончика пальца
        src_ind = np.power(src_out - res[-i], 2).sum(axis=1).argmin()
        res.append(src_out[src_ind])
        src_out = np.delete(src_out, src_ind, 0)

    return np.array(res)

def finger_line(filename):
    """
    Объединяем описанные выше функции в одну

    filename - имя изображения в рабочей директории
    """
    mask = get_mask(filename)
    coord = get_points(mask)

    return coord

#------------Основная часть программы-------------------

dots = []
labels = []
total = len(glob('./data/*.tif'))

start = time.time()

# Основной цикл обработки изображений
for count, filename in enumerate(glob('./data/*.tif')):
    print('Image processing: {}/{}'.format(count + 1, total))

    # Получаем "линию пальцев"
    line = finger_line(filename)

    # Сохраняем координаты точек и номер изображения
    dots.append(line)
    labels.append(int(filename.split('/')[-1][:-4]))

    # Сохраняем преобразованное изображение
    plt.imshow(Image.open(filename))
    plt.plot(line[:,1], line[:,0], c='r')
    plt.scatter(line[0][1], line[0][0], c='r')
    plt.axis('off')
    plt.savefig(os.path.join('./output', filename.split('/')[-1]), bbox_inches='tight')
    plt.close()

dists = np.zeros(8)

# Составляем признаковое описание ладоней
for line in dots:
    # Вычисляем длины отрезков ломаных
    dist = []
    for i in range(len(line) - 1):
        dist.append(np.linalg.norm(line[i] - line[i + 1]))

    # Если какие-то отрезки не нашлись, заменим их длины нулями
    if len(dist) < 8:
        dist += [0] * (8 - len(dist))
    
    dists = np.vstack([dists, np.array(dist)])

dists = dists[1:]

# Нули (повреждённые данные) заменим медианой по столбцу
for i in range(dists.shape[1]):
    dists[:,i][dists[:,i] == 0] = np.median(dists[:,i][dists[:,i] > 0])

print('Total time: {} s'.format(round(time.time() - start)))

# Сохраним полученное признаковое описание и метки объектов в файлы
labels = np.array(labels)

np.savetxt('./output/labels.csv', labels, delimiter=',')
np.savetxt('./output/dists.csv', dists, delimiter=',')