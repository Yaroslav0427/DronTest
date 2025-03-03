import numpy as np
from utils.elapsed import TimeElapsed


class PatchDetector:
    """
    Класс реализует детекцию на основе разделения на патчи.

    Args:
        image (np.ndarray): Входное изображение.
        crops (List): нарезанные фрагменты
        model: Предобученная модель, с помощью которой будем выполнять детекцию
        imgsz (tuple): Размер входного изображения для модели YOLO.
        conf (float): Порог уверенности для детекции YOLO.
        iou (float): IoU-порог для non-maximum suppression.
        classes_list (List[int] or None): Список классов, которые мы хотим детектировать.
                                          Если None, то все классы будут детектироваться
        time_elapse (TimeElapsed or None): объект для централизованного измерения длительности выполнения операций
        kvargs (dict): Дополнительные параметры (ключ-значение) которые можно пробросить с верхнего уровня. Например параметры для ultralytics
    """
    def __init__(
            self,
            image: np.ndarray,
            crops: list,
            model,
            model_img_size: tuple = (640, 640),
            conf=0.25,
            iou=0.7,
            classes_list=None,
            time_elapse: TimeElapsed = None,
            kvargs={},
    ) -> None:

        self.image = image  # входное изображение
        self.model = model  # модель, которой будем детектировать
        self.crops = crops  # нарезанные фрагменты
        self.model_img_size = model_img_size  # размер изображений, которые ожидает модель на вход
        self.conf = conf  # Порог уверенности для детекции
        self.iou = iou  # Порог IoU для non-maximum suppression
        self.classes_list = classes_list  # Классы для детекции

        self.time_elapse = time_elapse
        self.kvargs = kvargs

        self.class_names_dict = self.model.names  # словарь имён классов, которые поддерживает модель

        self._detect_objects()

    def _detect_objects(self):
        """
        Метод выполняет детекцию объектов на основе фрагментов

        Returns:
            None
        """
        self._calculate_inference()

        if self.time_elapse is not None:
            self.time_elapse.start("calculate_real_values")
        for crop in self.crops:
            crop.calculate_real_values()

        if self.time_elapse is not None:
            self.time_elapse.stop("calculate_real_values")

    def _calculate_inference(self):
        # Perform batch inference of image crops through a neural network

        # !!!!!! нужны батчи
        batch = [element.crop for element in self.crops]

        self.time_elapse.start("yolo_predict")
        predictions = self.model.predict(
            batch,
            imgsz=self.model_img_size,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes_list,
            verbose=False,
            **self.kvargs
        )
        self.time_elapse.stop("yolo_predict")

        self.time_elapse.start("processed bbox")
        for pred, crop in zip(predictions, self.crops):
            # Get the bounding boxes and convert them to a list of lists
            crop.detected_xyxy = pred.boxes.xyxy.cpu().int().tolist()

            # Get the classes and convert them to a list
            crop.detected_cls = pred.boxes.cls.cpu().int().tolist()

            # Get the mask confidence scores
            crop.detected_conf = pred.boxes.conf.cpu().numpy()

        self.time_elapse.stop("processed bbox")



