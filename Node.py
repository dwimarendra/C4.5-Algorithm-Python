class make_node(object):
    
    def __init__(self, set, parent, position):
        self.set = set
        self.position = position
        self.sepal_length = [] 
        self.sepal_width = [] 
        self.petal_length = []
        self.petal_width = []
        self.sepal_length_and_flower = [] 
        self.sepal_width_and_flower = []
        self.petal_length_and_flower = []
        self.petal_width_and_flower = []
        self.list_of_attribute_lists = []
        self.separate_attributes()
        self.combine_attributes_and_flower()
        self.children = []
        self.parent = parent
        
    def add_child(self, obj):
        self.children.append(obj)
        
    def isRoot(self):
        return not self.parent #returns true if parent is empty
    
    def isLeaf(self):
        return not self.children #returns true if it has no children

    def separate_attributes(self):
        for x in self.set:
            self.sepal_length.append("%.1f" % x[0]) 
            self.sepal_width.append("%.1f" % x[1])
            self.petal_length.append("%.1f" % x[2])
            self.petal_width.append("%.1f" % x[3])
        self.get_distinct_values()
        
    def combine_attributes_and_flower(self):
        for x in self.set:
            self.sepal_length_and_flower.append(("%.1f" % x[0], x[-1])) 
            self.sepal_width_and_flower.append(("%.1f" % x[1], x[-1]))
            self.petal_length_and_flower.append(("%.1f" % x[2], x[-1]))
            self.petal_width_and_flower.append(("%.1f" % x[3], x[-1]))        
    
    def get_distinct_values(self):
        self.sepal_length = list(set(self.sepal_length))
        self.sepal_width = list(set(self.sepal_width))
        self.petal_length = list(set(self.petal_length))
        self.petal_width = list(set(self.petal_length))
        self.sort_attributes()
            
    def sort_attributes(self):
        self.sepal_length.sort()
        self.sepal_width.sort()
        self.petal_length.sort()
        self.petal_width.sort()
        self.remove_last()
        
    def remove_last(self): 
    #Crucial for the C4.5 is that you do not want to take in consideration the highest value during threshold calculation
        if len(self.sepal_length) != 0:
            del self.sepal_length[-1]
        if len(self.sepal_width) != 0:
            del self.sepal_width[-1]
        if len(self.petal_length) != 0:
            del self.petal_length[-1] 
        if len(self.petal_length) != 0:
            del self.petal_width[-1]
        self.combine_separate_attributes()
        
    def combine_separate_attributes(self):
        self.list_of_attribute_lists.append(self.sepal_length)
        self.list_of_attribute_lists.append(self.sepal_width)
        self.list_of_attribute_lists.append(self.petal_length)
        self.list_of_attribute_lists.append(self.sepal_width)

        
        