
from termcolor import colored
import os
import shutil
import json

def save_statistic_data(file_name: str, data: dict):
    """ Сохраняет статистику по испытанию в общий файл json. 
        Дополнительно отслеживает возможные проблемы с защитой от полной утраты ранее полученных данных
        :param file_name: имя основного файла, куда нужно сохранять
        :param data: Данные в виде словаря. 
    """
    temp_file = "last_statistic_data.json"
    dir = os.path.dirname(file_name)
    file_only = os.path.basename(file_name)
    reserve_file_name = os.path.join(dir, f"_{file_only}")    # резервный файл, чтобы случайно не потерять предыдущие данные

    if os.path.exists(reserve_file_name):   # если резервный файл уже существует, то значит была такая-то ошибка во время предыдущего запуска.
        print(colored('Обнаружен резервный файл с информацией об предыдущих эксперименах.', 'red'))
        print(colored('Возможно сохранение статистики, которое было перед этим, не завершило успешно удаление.', 'red'))
        print(colored(f'Для восстановления работоспособности, нужно определить какой файл будем в дальнейшем использовать ({file_name, reserve_file_name}', 'red'))
        print(colored(f'Если нам нужно использовать файл {os.path.basename(reserve_file_name)}, то его нужно переименовать в {os.path.basename(file_name)}', 'red'))
        print(colored(f'После чего файл {os.path.basename(reserve_file_name)}, нужно удалить', 'red'))
        print(colored(f'Чтобы не потерять данные о текущем эксперименте, они будут сохранены в файл {temp_file}', 'red'))
        file_name = temp_file           # при такой проблеме будем сохранять во временный файл, он будет всегда перезаписываться.

    # Делаем резервную копию основного файла, но только если нет ошибки и мы не заменили имя файла для сохранения
    if os.path.exists(file_name) and file_name != temp_file:
        shutil.copy2(file_name, reserve_file_name)              # создаём резервную копию
        with open(file_name, 'r') as fp:                        # читаем содержимое файла (результаты других запусков)
            json_data = json.load(fp)
            json_data.append(data)                              # добавляем новые в имеющися данные
    else:
        json_data = [data]                                      # А если ещё нет исходного файла, то формируем список, в котором только один элемент - данные для сохранения
    

    with open(file_name, 'w') as fp:                            # Сохраняем данные в файл
        json.dump(json_data,fp)
    
    if os.path.exists(reserve_file_name) and file_name != temp_file:        # если ранее создавали резервный файл, то удаляем его.
        os.remove(reserve_file_name)