
import time

class TimeElapsed:
    def __init__(self):
        self.results = dict()
        self.timers = dict()


    def start(self, name: str):
        assert name not in self.timers.keys(), "Таймер уже был запущен ранее и не остановлен, попытка повторного запуска похожа на ошибку." 
        self.timers[name] = time.time()


    def stop(self, name: str):
        t = time.time()
        assert name in self.timers, "Неизвестное имя, вероятно с таким именем не был запущен отсчёт (смотри метод start_measurement)"
        self.add(name, t - self.timers[name])

        del self.timers[name]
    

    def add(self, name:str, elapsed:float):
        if name not in self.results.keys():
            self.results[name] = elapsed
        else:
            self.results[name] += elapsed


    def __getitem__(self, key):
        return self.results[key]
    
    def __iter__(self):
        return iter(self.results)
    
    def __repr__(self):
        return repr(self.results)

    
