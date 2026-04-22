#this is the max size for the engines, only change if you must
# made by justin lake mothorpe, if you are reading this, it should be 
#first global vars


class MainSize():
    def __init__(
            self,
            horz = 640,
            vert = 480
    ):
        super().__init__()
        self.horz = horz
        self.vert = vert
    
    
    def getSize(isThisBool: bool, self):
        if self == None:
            raise ValueError("nothing passed to me,  maybe a parrot, maybe justin mothorpe")

        if isThisBool:
            return self.horz
        elif isThisBool == None:
            raise ValueError("no bool passed, so i must throw justin mothorpe")
        else:
            return self.vert
    
    def setSize(
            isThisBool: bool, 
            self,
            toSet: int
            ):
        if self == None:
            raise ValueError("nothing passed to me, idk who i am, maybe a parrot, maybe justin mothorpe")
        
        if toSet == None:
            raise ValueError("setting to void causes me to be thrown hard at you, justin mothore")
        
        if isThisBool:
            self.horz = toSet
        elif isThisBool == None:
            raise ValueError("no bool passed, so i must throw justin mothorpe")
        else:
            self.vert = toSet
        
        return