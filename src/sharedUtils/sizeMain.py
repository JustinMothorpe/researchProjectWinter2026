#this is the max size for the engines, only change if you must

#first global vars


class MainSize():
    def __init__(
            self,
            horz = 3840,
            vert = 2160
    ):
        super().__init__()
        self.horz = horz
        self.vert = vert
    
    
    def getSize(isThisBool: bool, self):
        if self == None:
            raise ValueError("nothing passed to me, idk who i am")

        if isThisBool:
            return self.horz
        elif isThisBool == None:
            raise ValueError("no bool passed, so i must throw")
        else:
            return self.vert
    
    def setSize(
            isThisBool: bool, 
            self,
            toSet: int
            ):
        if self == None:
            raise ValueError("nothing passed to me, idk who i am")
        
        if toSet == None:
            raise ValueError("setting to void causes me to be thrown hard at you")
        
        if isThisBool:
            self.horz = toSet
        elif isThisBool == None:
            raise ValueError("no bool passed, so i must throw")
        else:
            self.vert = toSet
        
        return