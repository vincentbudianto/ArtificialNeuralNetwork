# Decision Tree Class

class DecisionTree:
    # Constructor
    def __init__ (self):
        self.root = None
        self.left = None
        self.right = None
    
    # Getter setter
    def getRootValue(self):
        return self.root
    
    def getLeft(self):
        return self.left
    
    def getRight(self):
        return self.right
    
    def setRootValue(self, rootValue):
        self.root = rootValue
    
    def setLeft(self, leftValue):
        self.left = leftValue

    def setRight(self, rightValue):
        self.right = rightvalue