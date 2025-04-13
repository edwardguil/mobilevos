# mobilevos/modules/memory.py
class MemoryBank:
    def __init__(self, max_length):
        self.max_length = max_length
        self.keys = []
        self.values = []
    
    def push(self, key, value):
        self.keys.append(key)
        self.values.append(value)
        if len(self.keys) > self.max_length:
            self.keys.pop(0)
            self.values.pop(0)
    
    def get_all(self):
        return self.keys, self.values
    
    def reset(self):
        self.keys = []
        self.values = []