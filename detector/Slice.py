

class Slice():

    def __init__(
            self, 
            x1: int,
            y1: int,
            x2: int,
            y2: int,
            row: int,
            col: int):
        
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.row = row
        self.col = col

    # def __repr__(self):
    #     return image_full[self.y1:self.y2, self.x1:self.x2, :]