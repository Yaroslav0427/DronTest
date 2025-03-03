
import os
from pillow_heif import register_heif_opener
register_heif_opener()

IMAGE_EXTENSIONS = {'.tiff', '.tif', '.jpeg', '.jpg', '.jpe', '.bmp', '.png','.heic', '.webp', '.jfif', '.psd', '.dng', '.nef'}

def is_image_file(filename:str):
    """
    Проверяем, что файл является изображением. 
    В данный момент протестированы только данные форматы файлов изображений.
    :param filename: полный путь к файлу
    :return: True - если файл является изображением
    """
    _, file_extension = os.path.splitext(filename)
    return file_extension in IMAGE_EXTENSIONS


def calculate_count_file_extensions(folder: str):
    """
    Функция возвращает словарь, где в качестве ключа выступает расширение файла, а в качестве значение количество файлов с таким расширением
    :param folder: папка, для которой будут искать все возможные расширения файлов, включая вложенные
    :return: Возвращает словарь, где в качестве ключа выступают расширения, а в качестве значения - количества
    """
    
    exts = dict()
    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_dir():
                sub_exts = calculate_count_file_extensions(entry.path)
                for ext_key in sub_exts:
                    if ext_key not in exts.keys():
                        exts[ext_key] = 0
                    exts[ext_key] += sub_exts[ext_key]
                
            elif entry.is_file():
                ext_key = str.lower(os.path.splitext(entry.name)[1])
                if ext_key not in exts.keys():
                    exts[ext_key] = 0
                exts[ext_key] += 1
    
    return exts     # Возвращаем набор расширений всех файлов в этой папке


def get_count_image_files(folder: str):
    """
    Определяет, сколько файлов изображения содержит папка.
    :param folder: имя папки, для которой будет подсчитано количество файлов, в том числе и во вложенных папках
    :return: Возвращает количество файлов изображений внутри папки, включая и вложенные.
    """
    return len(get_all_image_file_names_in_folder(folder))   # получаем имена файлов изображений и папок, внутри папки folder


def get_all_image_file_names_in_folder(folder:str):
    """
    Возвращает список всех имён (путей) файлов изображений, находящихся в папке folder, а тажке во всех вложенных на всех уровнях.
    Рекурсивная функция
    :param folder: имя папки верхнего уровня, для которой нужно найти все вложенные файлы изображений, включая и во вложенных файлах
    :return: Возвращает список путей к файлам изображений, которые находятся внутри папки folder на всех вложенных уровнях.
    """
    tuple_extentions = tuple(IMAGE_EXTENSIONS)
    return get_all_files_with_specific_extentions(folder, tuple_extentions)
    

def get_all_files_with_specific_extentions(folder:str, extentions: tuple):
    """
    Возвращает список всех имён (путей) файлов изображений, находящихся в папке folder, а тажке во всех вложенных на всех уровнях.
    Рекурсивная функция
    :param folder: имя папки верхнего уровня, для которой нужно найти все вложенные файлы изображений, включая и во вложенных файлах
    :extentions: список расширений файлов, которые нужно найти, например [".jpg", ".jpeg"]
    :return: Возвращает список путей к файлам изображений, которые находятся внутри папки folder на всех вложенных уровнях.
    """
    result_list = list()
    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_dir():
                result_list.extend(get_all_files_with_specific_extentions(entry.path, extentions))
            elif str.lower(entry.name).endswith(extentions) and entry.is_file():
                result_list.append(entry.path)

    return result_list


def get_all_files(folder:str):
    """
    Возвращает список всех имён (путей) файлов находящихся в папке folder, а тажке во всех вложенных на всех уровнях.
    Рекурсивная функция
    :param folder: имя папки верхнего уровня, для которой нужно найти все вложенные файлы, включая и во вложенных файлах
    :return: Возвращает список путей к файлам, которые находятся внутри папки folder на всех вложенных уровнях.
    """
    result_list = list()
    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_dir():
                result_list.extend(get_all_files(entry.path))
            elif entry.is_file():
                result_list.append(entry.path)

    return result_list


def get_file_name_only(file_path: str):
    return file_path.split(os.sep)[-1][::-1].split('.', 1)[1][::-1]


if __name__ == "__main__":
    folders = [
        'f:/Arma_dataset',
        
    ]
    file_extentions_dict = dict()
    
    for folder in folders:
        print(f"Folder: {folder}")
        print(f"count image files: {get_count_image_files(folder)}")
        
        for key, value in calculate_count_file_extensions(folder).items():
            if key in file_extentions_dict.keys():
                file_extentions_dict[key] += value
            else:
                file_extentions_dict[key] = value

    print(f"file extentions: {file_extentions_dict.keys()}")

    ext_sorted = dict(sorted(file_extentions_dict.items(), key=lambda item: item[1], reverse=True))

    print(42*"-")
    for ext_key in ext_sorted:
        # print(f"| {ext_key:30} | {file_extentions_dict:10d} |")
        print(f"| {ext_key:25} | {ext_sorted[ext_key]:10} |")
    print(42*"-")