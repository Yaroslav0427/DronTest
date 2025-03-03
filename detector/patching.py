""" Автор: Мотькин Игорь Сергеевич. 
"""

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from random import randint
from detector.SliceElement import SliceElement
from detector.Slice import Slice

def split_matrix(
    img_width,
    img_height,
    size_hor: int,
    size_ver: int,
    overlap_hor_prc=15,
    overlap_ver_prc=15,
):
    
    # --- уточняем размер перекрытия, чтобы обработать все части изображения ----
    # Получаем на сколько частей нужно разбить изображение
    steps_hor = 1 + (img_width - size_hor) / (size_hor * (1 - (overlap_hor_prc / 100)))
    steps_ver = 1 + (img_height - size_ver) / (size_ver * (1 - (overlap_ver_prc / 100)))

    # округляем количество разбиений до целого вверх, Это заставит немного увеличить перекрытие
    steps_hor = math.ceil(steps_hor)
    steps_ver = math.ceil(steps_ver)

    cross_koef_hor = (img_width - size_hor) / (size_hor * (steps_hor - 1))
    cross_koef_ver = (img_height - size_ver) / (size_ver * (steps_ver - 1))

    assert 1 + (img_width - size_hor) / (size_hor * cross_koef_hor) == steps_hor
    assert 1 + (img_height - size_ver) / (size_ver * cross_koef_ver) == steps_ver

    slices = list()
    for j in range(steps_ver):
        for i in range(steps_hor):

            slices.append(Slice(
                                x1 = int(size_hor * i * cross_koef_hor),
                                y1 = int(size_ver * j * cross_koef_ver),
                                x2 = int(size_hor * i * cross_koef_hor + size_hor), 
                                y2 = int(size_ver * j * cross_koef_ver + size_ver),
                                col=i,
                                row=j))

    return slices

def get_slices(
    image_full,
    target_slice_width: int,
    target_slice_height: int,
    sizes_hor: list = None,
    sizes_ver: list = None,
    overlap_hor_prc=15,
    overlap_ver_prc=15,
):
    if sizes_hor is None:
        sizes_hor = [target_slice_width]
    if sizes_ver is None:
        sizes_ver = [target_slice_height]

    assert len(sizes_hor) == len(sizes_ver), "списки размеров должны иметь одинаковою длину"

    levels = list()

    img_height, img_width = image_full.shape[:2]

    for size_hor, size_ver in zip(sizes_hor, sizes_ver):
        levels.append(
            split_matrix(
                img_width,
                img_height,
                size_hor=size_hor,
                size_ver=size_ver,
                overlap_hor_prc=overlap_hor_prc,
                overlap_ver_prc=overlap_ver_prc
        ))
    
    data_all_crops = list()
    for slices in levels:
        for slice in slices:
            im_temp = image_full[slice.y1:slice.y2, slice.x1:slice.x2, :]

            data_all_crops.append(SliceElement(
                source_image=image_full,
                crop=im_temp,
                x_start=slice.x1,
                y_start=slice.y1,
            ))

    return levels, data_all_crops
    

def draw_separation_lines(image, levels:list, thickness:int=5, path_to_save: str=None):

    painted_images = list()
    # для каждого уровня будет отдельная демонстрация разделения
    for level in levels:
        img_copy = image.copy()

        for idx, slice in enumerate(level):
            color = (randint(0,255), randint(0,255), randint(0,255))
            cv2.rectangle(img_copy, (slice.x1, slice.y1), (slice.x2, slice.y2), color=color, thickness=thickness)
            cv2.putText(img_copy, f"{idx+1}", org=(slice.x1 + (slice.x2 - slice.x1)//2, slice.y1 + (slice.y2 - slice.y1)//2), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=color, thickness=5)
        painted_images.append(img_copy)

    if path_to_save is not None:
        os.makedirs(path_to_save, exist_ok=True)            
        for idx, image in enumerate(painted_images):
            cv2.imwrite(os.path.join(path_to_save,f"split_lines_{idx}.png"), img_copy)
            
    return painted_images


def save_cropped_images(image, levels:list, path_to_save:str):
    for idx, slices in enumerate(levels):
        result_path = os.path.join(path_to_save, f"level-{idx}")
        os.makedirs(result_path, exist_ok=True) 

        for slice in slices:
            file_path = os.path.join(result_path, f"slice_{slice.row:02d}_{slice.col:02d}.png")
            cropped_img = image[slice.y1:slice.y2, slice.x1:slice.x2, :]
            cv2.imwrite(file_path, cropped_img)
