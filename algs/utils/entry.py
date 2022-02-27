class LRU_Entry:
    def __init__(self, id, size):
        self.id = id
        self.size = size

    def __repr__(self):
        return "(id={}, size={})".format(self.id, self.size)
