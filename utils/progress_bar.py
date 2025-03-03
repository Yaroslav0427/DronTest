

import time

class ProgressBar:
    """ Прогресс бар для длительных операций
    """
    def __init__(self, count_items: int, title: str = "Обработано: ", length_bar: str = 100):
        """ Инициализируем объект

        Args:
            count_items (int): Количество элементов в 100%
            title (_type_, optional): Текст перед прогресс баром. По умолчанию "Обработано: ".
            length_bar (str, optional): Длина бара. По умолчанию 100 символов.
        """
        self.count_items = count_items
        self.title = title
        self.length_bar = length_bar
        self.count_1item = count_items / 100        # количество элементов в 1%
        self.len_1pixel = 100 / length_bar          # количество процентов на 1 символ прогресса
        self.animation_symbols = ['\\','|','/', '-', '\\', '|', '-']    # порядок символов для анимации
        self.animation_symbol_idx = 0       # текущий символ для анимации
        self.animation_old_time = time.time()           # какое было на момент предыдущего изменения символа анимации
        self.animation_interval = 1.0 / 5     # знаметатель - это приблизительная частата смены символа в Герцах (5 -> это 5 изменений в секунду (приблизительное))
        self.Running = True                     # то, что бар работает 
    
    def update(self,idx: int):
        """ На каждой итерации цикла вызываем этот метод и передаём номер текущего элемента

        Args:
            idx (ing): номер текущего элемента
        """
        should_change_data = int((idx+1) % self.count_1item) == 0 or idx == 0 or idx == self.count_items - 1   # нужно ли изменять основные данные
        should_change_animation = False         # нужно ли перерисовывать символ анимации
        new_time = time.time()      # получаем текущее время (для анимации)
        if new_time - self.animation_old_time >= self.animation_interval or should_change_data:    # нужно ли менять символ анимацуии
            self._shift_animation_symbol()              # меняем символ анимации
            should_change_animation = True              # указываем, что нужно перерисовать символ анимации
            self.animation_old_time = new_time          # сохраняем текущее время, чтобы уже относительно его делать анимацию
        
        animation = f"   {self.animation_symbols[self.animation_symbol_idx]}  "     # строка отвечающая за анимацию. В основном содержит отступы и нужный символ анимации
        if should_change_data:                                              # если же нужно обновить все данные в прогресс баре
            percent = int(round((idx+1) / self.count_1item))            # высчитываем текущий процент (откругляем до большего значения)
            progress_bar = f"[{int(percent // self.len_1pixel) * '■'}"                          # прорисовываем прогресс бар
            progress_bar += f"{int(self.length_bar - (percent // self.len_1pixel)) * ' '}] "
            progress_bar += f"{percent}"                                                        # добавляем ему значение процентов
            print(f"{animation} {self.title} {progress_bar}", end="%\r")                        # отображаем на экране полный текст
        elif should_change_animation:                                       # если же нужно изменить только символ анимации, то
            print(f"{animation}", end="\r")                                 # перерысовываем его (он специально в самом начале стоит)
        
        if idx == self.count_items - 1:                                     # если мы достигли последнего элемента
            print()                                                     # то нужно перейти на новую строку, чтобы следующая команда не затёрла текст
    
    
    def _shift_animation_symbol(self):                                      # функция выполняем переключение индекса символа анимации попорядку
        if self.animation_symbol_idx + 1 == len(self.animation_symbols):
            self.animation_symbol_idx = 0
        else:
            self.animation_symbol_idx += 1