# Decision Tree Class

# Kemungkinan yang dapat direvisi:
# Menambahkan atribut entropy dan jumlah sample yang masuk
# Mungkin data sampel yang masuk
# Mungkin string yang dieksekusi buat tree (eval(<string>))

class DecisionTree:
    # Constructor
    def __init__(self, value = None, attribute = None):
        self.attribute = attribute
        self.root = value
        self.nodes = []

    # Getter setter
    def getAttributeValue(self):
        return self.attribute

    def getRootValue(self):
        return self.root

    def getNodes(self):
        return self.nodes

    def setAttributeValue(self, attributeValue):
        self.attribute = attributeValue

    def setRootValue(self, rootValue):
        self.root = rootValue

    def setNodes(self, nodesValue):
        self.nodes.append(nodesValue)

    def setNodesValue(self, nodesValue):
        self.nodes.append(DecisionTree(nodesValue))

    # Print tree without visualization library (GraphViz)
    def printTree(self, tabCounter = 0):
        if self.root is not None:
            if (tabCounter > 0):
                for i in range(tabCounter - 1):
                    print("|  ", end = "")
                print(self.attribute)
            for i in range(tabCounter):
                print("|  ", end = "")
            print(self.root)
            if len(self.nodes) != 0:
                for node in self.nodes:
                    node.printTree(tabCounter + 1)

    # Pass through tree
    def classifyDatum(self, datum):
        print(self.attribute)

# Test data
# tree = DecisionTree()
# tree.setRootValue("test")
# leftTree = DecisionTree()
# leftTree.setRootValue("testLeft")

# rightTree = DecisionTree()
# rightTree.setRootValue("testRight")

# rightLeftTree = DecisionTree()
# rightLeftTree.setRootValue("testRightLeft")

# rightRightTree = DecisionTree()
# rightRightTree.setRootValue("testRightRight")

# tree.setLeft(leftTree)
# tree.setRight(rightTree)
# rightTree.setLeft(rightLeftTree)
# rightTree.setRight(rightRightTree)
# leftTree.setLeftValue("testLeftLeft")
# leftTree.setRightValue("testLeftRight")

# tree.printTree()