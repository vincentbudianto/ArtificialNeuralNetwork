'''
Layer class
Handles information
- Node_count        : the number of nodes in the said layer
- LayerNo           : the level of the layer (0 being the starting layer)
- IsStart           : boolean (if the layer is the starting layer)
- Weight            : list of list ([current_layer.node_count][previouslayer.node_count])
- ActivationType    : the type of activation function of the layer (like sigmoid)
- Value             : the value of the object (list)
- Delta             : the value of the delta of a node (list)
- DeltaWeight       : temporary value that will be flushed after every epoch
'''

class Layer:
    def __init__(self, node_count, layerNo, activationType):
        self.node_count = node_count
        self.layerNo = layerNo
        self.isStart = (layerNo == 0)
        self.activationType = activationType
        self.weight = None
        self.value = None
        self.delta = None
        self.deltaWeight = None