import argparse
import cv2
import json
import math
import matplotlib.pyplot as plt
import os
import re
from tqdm import tqdm
import subprocess
from termcolor import colored
from ultralytics import YOLO

from detector.CombineDetections import CombineDetections
from detector.patching import get_slices, draw_separation_lines, save_cropped_images, split_matrix
from detector.PatchDetector import PatchDetector
from detector.visualizing import visualize_results
from utils.core import get_all_image_file_names_in_folder
from utils.elapsed import TimeElapsed
from utils.progress_bar import ProgressBar
from utils.statistics import save_statistic_data

def extract_resolution_from_model_name(model_name):
    """
    Извлекает разрешение из названия модели (например, yolov8_640 -> 640).
    Если разрешение не найдено, возвращает значение по умолчанию 640.
    """
    match = re.search(r"_(\d+)\.pt$", model_name)  # Ищем число перед .pt
    if match:
        return int(match.group(1))
    return 640  # Значение по умолчанию


def load_existing_statistics(output_path):
    """
    Загружает существующие данные из statistics.json, если файл существует и не пустой.
    :param output_path: Путь к папке с файлом statistics.json.
    :return: Список с существующими отчетами или пустой список, если файл не существует или пустой.
    """
    statistics_file = os.path.join(output_path, "statistics.json")
    if os.path.exists(statistics_file):
        try:
            with open(statistics_file, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():  # Проверяем, что файл не пустой
                    return json.loads(content)
        except json.JSONDecodeError:
            print(colored(f"Файл {statistics_file} содержит некорректные данные. Будет создан новый файл.", 'red'))
    return []  # Возвращаем пустой список, если файл не существует или пустой


def save_statistic_data(output_path, report):
    """
    Сохраняет отчеты в файл statistics.json.
    :param output_path: Путь к папке, где будет сохранен файл.
    :param report: Отчет для сохранения.
    """
    statistics_file = os.path.join(output_path, "statistics.json")
    reports = load_existing_statistics(output_path)  # Загружаем существующие данные
    reports.append(report)  # Добавляем новый отчет
    with open(statistics_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=4, ensure_ascii=False)


def process_video(models_dir, video_file, overlap_hor=15, overlap_ver=15):
    """
    Основная функция для обработки видео с использованием моделей из папки.
    :param models_dir: Папка с моделями.
    :param video_file: Путь к видео файлу.
    :param overlap_hor: Процент перекрытия по горизонтали.
    :param overlap_ver: Процент перекрытия по вертикали.
    """
    # Получаем список всех моделей в папке
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pt")]  # Ищем файлы с расширением .pt
    if not model_files:
        print(colored(f"В папке {models_dir} нет моделей (.pt файлов).", 'red'))
        return

    print(colored(f"Найдено моделей: {len(model_files)}", 'green'))

    # Определяем output_path
    output_path = "artifacts"  # Или любой другой путь
    os.makedirs(output_path, exist_ok=True)  # Создаем папку, если она не существует

    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        print(colored(f"\nОбработка модели: {model_file}", 'blue'))

        # Извлекаем разрешение из названия модели
        resolution = extract_resolution_from_model_name(model_file)
        model_width = resolution
        model_height = resolution
        slice_width = resolution
        slice_height = resolution

        # Выводим параметры модели и входное разрешение YOLO
        print(colored("\nПараметры модели и входное разрешение YOLO:", 'yellow'))
        print("-" * 50)
        print(f"| {'model_width':>20} | {model_width:<25} |")
        print(f"| {'model_height':>20} | {model_height:<25} |")
        print(f"| {'slice_width':>20} | {slice_width:<25} |")
        print(f"| {'slice_height':>20} | {slice_height:<25} |")
        print(f"| {'YOLO Input Width':>20} | {model_width:<25} |")
        print(f"| {'YOLO Input Height':>20} | {model_height:<25} |")
        print("-" * 50)

        # Открываем видео
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print(colored("Ошибка открытия файла", 'red'))
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = round(frame_count / fps, 2)

        print(f"fps: {fps:.0f}")
        print(f"Ширина: {frame_width}")
        print(f"Высота: {frame_height}")
        print(f"Количество кадров: {frame_count}")
        print(f"Длительность (сек): {video_duration}")

        slice_count = len(split_matrix(frame_width, frame_height, slice_width, slice_height, overlap_hor, overlap_ver))
        print(f"Количество патчей: {slice_count}")

        model = YOLO(model_path, task="detect")

        sizes_hor = [slice_width]
        sizes_ver = [slice_height]

        time_elapse = TimeElapsed()

        # Используем tqdm для отображения прогресса
        progress_bar = tqdm(total=frame_count, desc="Обработка кадров", unit="кадр")

        idx = 0
        time_elapse.start("-= Total =-")

        while True:
            time_elapse.start("read_frame")
            ret, frame = cap.read()
            time_elapse.stop("read_frame")

            if not ret:
                break

            time_elapse.start("make_slices")
            levels, crops = get_slices(
                frame,
                target_slice_width=model_width,
                target_slice_height=model_height,
                sizes_hor=sizes_hor,
                sizes_ver=sizes_ver,
                overlap_hor_prc=overlap_hor,
                overlap_ver_prc=overlap_ver,
            )
            time_elapse.stop("make_slices")

            element_crops = PatchDetector(
                image=frame,
                crops=crops,
                model=model,
                model_img_size=(model_width, model_height),
                conf=0.8,
                iou=0.7,
                classes_list=[0],  # , 1, 2, 3, 5, 7],
                time_elapse=time_elapse,
            )

            # Убираем измерение времени для combine_detections
            CombineDetections(element_crops, nms_threshold=0.05)

            idx += 1
            progress_bar.update(1)  # Обновляем прогресс-бар

        cap.release()
        cv2.destroyAllWindows()

        progress_bar.close()  # Закрываем прогресс-бар

        time_elapse.stop("-= Total =-")

        # Рассчитываем фактический FPS
        actual_fps = frame_count / time_elapse["-= Total =-"]

        print(colored("\nТребуемое время:", 'yellow'))
        print("-" * 50)
        print(f"|                Имя                | {' ' * 3} сек {' ' * 3}|")
        print("-" * 50)
        for key in time_elapse:
            if key != "combine_detections":  # Исключаем combine_detections из вывода
                print(f"| {key:>33} | {time_elapse[key]:>10.2f} |")
        print("-" * 50)

        # Заполняем словарь отчёта
        report = {
            "file_name": os.path.basename(video_file),
            "fps": int(round(fps, 0)),
            "actual_fps": round(actual_fps, 2),  # Фактический FPS
            "original_video_size": (frame_width, frame_height),
            "video_duration_sec": video_duration,
            "model_name": os.path.basename(model.model_name),
            "yolo_input_resolution": (model_width, model_height),  # Добавляем входное разрешение YOLO
            "patch_size": (slice_width, slice_height),
            "patch_overlap_prc": (overlap_hor, overlap_ver),
            "patch_count": slice_count,
            "yolo_predict_time": round(time_elapse["yolo_predict"], 2),
            "total_frame_processing_time": round(time_elapse["-= Total =-"], 2),  # Общее время обработки кадра
        }

        # Итоговая статистика
        print(colored("\nИтоговая статистика:", 'yellow'))
        for key, value in report.items():
            print(f"{key:>30} | {value}")

        # Сохраняем отчет в файл
        save_statistic_data(output_path, report)

        print(f"Отчет для модели {model_file} сохранен в statistics.json.\n")

    print("Все модели обработаны. Программа завершена.\n")


# Пример вызова функции
process_video(
    models_dir=r"C:\Users\Legion\PycharmProjects\DronTest\models",  # Папка с моделями
    video_file=r"C:\Users\Legion\PycharmProjects\DronTest\demo\videos\3840_2160.mp4",
    overlap_hor=15,
    overlap_ver=15,
)