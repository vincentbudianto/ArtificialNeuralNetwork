# Decision Tree Class

# Kemungkinan yang dapat direvisi:
# Menambahkan atribut entropy dan jumlah sample yang masuk
# Mungkin data sampel yang masuk
# Mungkin string yang dieksekusi buat tree (eval(<string>))

class DecisionTree:
    # Constructor
    def __init__(self, value = None):
        self.root = value
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
        self.right = rightValue

    def setLeftValue(self, leftValue):
        self.left = DecisionTree(leftValue)

    def setRightValue(self, rightValue):
        self.right = DecisionTree(rightValue)

    # Print tree without visualization library (GraphViz)
    def printTree(self, tabCounter = 0):
        if self.root is not None:
            for i in range(tabCounter):
                print("|  ", end = "")
            print(self.root)
            if self.left is not None:
                self.getLeft().printTree(tabCounter + 1)
            if self.right is not None:
                self.getRight().printTree(tabCounter + 1)


# Test data
tree = DecisionTree()
tree.setRootValue("test")

leftTree = DecisionTree()
leftTree.setRootValue("testLeft")

rightTree = DecisionTree()
rightTree.setRootValue("testRight")

rightLeftTree = DecisionTree()
rightLeftTree.setRootValue("testRightLeft")

rightRightTree = DecisionTree()
rightRightTree.setRootValue("testRightRight")

tree.setLeft(leftTree)
tree.setRight(rightTree)
rightTree.setLeft(rightLeftTree)
rightTree.setRight(rightRightTree)
leftTree.setLeftValue("testLeftLeft")
leftTree.setRightValue("testLeftRight")

tree.printTree()