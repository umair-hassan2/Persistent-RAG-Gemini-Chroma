print("hey hello world")

class Sys:
    def __init__(self,x,y)->None:
        self.x = x
        self.y = y
    
    def get_cross(self):
        return self.x * self.y
    
    def get_dot(self):
        return self.x * self.y
    
    def double(self):
        self.x = self.x * 2
        return self.x
    