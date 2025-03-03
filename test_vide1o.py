import argparse
import cv2
import math
import matplotlib.pyplot as plt
import os
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



# Задаём поддерживаемые аргументы командной строки
parser = argparse.ArgumentParser(description='Детекция дронов на видео')
parser.add_argument('--model-dir', type=str, metavar='PATH',
                    help='Путь к предобученной модели')
parser.add_argument('--video-file', type=str, metavar='PATH',
                    help='Путь к видео файлу, который нужно обработать')

parser.add_argument('--model-width', default=640, type=int, 
                    help='Ширина изображения с которым работает предобученная модель')
parser.add_argument('--model-height', default=640, type=int,
                    help='Высота изображения с которым работает предобученная модель')

parser.add_argument('--slice-width', default=640, type=int,
                    help='Ширина фгарментов на которые будет нарезано входное изображение (кадр)')
parser.add_argument('--slice-height', default=640, type=int,
                    help='Высота фгарментов на которые будет нарезано входное изображение (кадр)')

parser.add_argument('--overlap-hor', default=15, type=int,
                    help='Процент перекрытия по горизонтали во время нарезки (относительно параметра slice-width)')
parser.add_argument('--overlap-ver', default=15, type=int,
                    help='Процент перекрытия по вертикали во время нарезки (относительно параметра slice-height)')

parser.add_argument('--should-save-cropped-images', action='store_true', default=False,
                    help='Нужно ли сохранять как выполняется нарезка')
parser.add_argument('--should-fix-output-file', action='store_true', default=False,
                    help='Нужно ли с помощью ffmpeg переконвертировать выходной файл, например чтобы он открывался в WSL (требует наличия ffmpeg)')

args = parser.parse_args()


print(colored("\nАргументы:", 'yellow'))
print("-" * 145)
print(f"|               Имя                | {' ' * 48} Значение {' ' * 49}|")
print("-" * 145)
for arg, value in vars(args).items():
    print(f"| {arg:>32} | {value:<106} |")
print("-" * 145)


output_path = os.path.join("artifacts", "video")
output_video_file = os.path.join(output_path, f"{os.path.splitext(os.path.basename(args.video_file))[0]}.mp4")
if not os.path.exists(output_path):
    os.makedirs(output_path)

cap = cv2.VideoCapture(args.video_file)

if not cap.isOpened():
    print(colored("Ошибка открытия файла", 'red'))
else:
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

    slice_count = len(split_matrix(frame_width, frame_height, args.slice_width, args.slice_height, args.overlap_hor, args.overlap_ver))
    print(f"Количество патчей: {slice_count}")

    model = YOLO(args.model_dir, task="detect")

    sizes_hor = [args.slice_width]
    sizes_ver = [args.slice_height]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    time_elapse = TimeElapsed()

    progress = ProgressBar(frame_count)
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
            target_slice_width=args.model_width,
            target_slice_height=args.model_height,
            sizes_hor=sizes_hor,
            sizes_ver=sizes_ver,
            overlap_hor_prc=args.overlap_hor,
            overlap_ver_prc=args.overlap_ver,
        )
        time_elapse.stop("make_slices")

        if args.should_save_cropped_images and idx == 0:
            time_elapse.start("save_slices")
            separation_line_images = draw_separation_lines(frame, levels, thickness=5, path_to_save="artifacts/images")
            save_cropped_images(frame, levels, path_to_save="artifacts/images")
            time_elapse.stop("save_slices")


        element_crops = PatchDetector(
            image=frame,
            crops = crops,
            model = model,
            model_img_size= (args.model_width, args.model_height),
            conf=0.8,
            iou=0.7,
            classes_list=[0], #, 1, 2, 3, 5, 7],
            time_elapse=time_elapse,
        )

        time_elapse.start("combine_detections")
        result = CombineDetections(element_crops, nms_threshold=0.05)
        time_elapse.stop("combine_detections")

        time_elapse.start("draw_to_image")
        img_painted = visualize_results(
            img=result.image,
            confidences=result.filtered_confidences,
            boxes=result.filtered_boxes,
            classes_ids=result.filtered_classes_id,
            classes_names=result.filtered_classes_names,
            thickness=8,
            show_boxes=True,
            show_class=False,
            axis_off=False,
            return_image_array=True
        )
        time_elapse.stop("draw_to_image")

        time_elapse.start("save_to_video")
        out.write(img_painted)
        time_elapse.stop("save_to_video")

        idx += 1
        progress.update(idx)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    
    time_elapse.stop("-= Total =-")

    print(colored("\nТребуемое время:", 'yellow'))
    print("-" * 50)
    print(f"|                Имя                | {' ' * 3} сек {' ' * 3}|")
    print("-" * 50)
    for key in time_elapse:
        print(f"| {key:>33} | {time_elapse[key]:>10.2f} |")
    print("-" * 50)

    # заполняем словарь отчёта
    report = {
        "file_name": os.path.basename(args.video_file),
        "fps": int(round(fps,0)),
        "original_video_size": (frame_width, frame_height),
        "video_duration_sec": video_duration,
        "model_name": os.path.basename(model.model_name),
        "patch_size": (args.slice_width, args.slice_height),
        "patch_overlap_prc": (args.overlap_hor, args.overlap_ver),
        "patch_count": slice_count,
        "yolo_predict_time": round(time_elapse["yolo_predict"], 2),
        "productivity_ratio": round(video_duration / time_elapse["yolo_predict"], 2)
    }
    report["max_possible_FPS"] = int(round(report["fps"] * report["productivity_ratio"], 0))

    print(colored("\nИтоговая статистика:", 'yellow'))
    for key, value in report.items():
        print(f"{key:>20} | {value}")

    save_statistic_data("statistics.json", report)

    # исправляем видео файл, чтобы можно было смотреть в WSL    
    if args.should_fix_output_file:
        print("\nПреобразуем выходной файл")
        left_part, ext = os.path.splitext(output_video_file)
        output_video_file_temp = f"{left_part}_temp{ext}"
        os.rename(output_video_file, output_video_file_temp)
        subprocess.run([f"ffmpeg", "-loglevel", "quiet", "-y", "-i", f"{output_video_file_temp}", "-c:v", "libx264", "-pix_fmt", "yuv420p", f"{output_video_file}"])
        os.remove(output_video_file_temp)


print("Программа завершена\n")
