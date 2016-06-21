class tree_node(object):
    #tree_node is a class for nodes that contain information on the structure of the final learned C4.5 decision tree classifier.
    
    def __init__(self, parent, leftchildren, rightchilden, threshold, attribute, depth, output):
        self.set = set
        self.leftchildren = []
        self.rightchildren = []
        self.threshold = threshold
        self.attribute = attribute
        self.depth = depth
        self.output = output
        self.parent = parent
        
        
    def add_left(self, obj):
        self.leftchildren.append(obj)
        
    def add_right(self, obj):
        self.rightchildren.append(obj)
        
        