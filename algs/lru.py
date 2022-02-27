from utils.dequedict import DequeDict
from utils.cacheop import CacheOp
from utils.progress_bar import ProgressBar
from utils.entry import LRU_Entry
from copy import deepcopy


class LRU:

    def __init__(self, cache_size, reader=None, rt_seg_window=0, **kwargs):
        self.name = "lru"
        self.cache_size = cache_size
        self.reader = reader
        self.current_size = 0
        self.lru = DequeDict()

        self.misses = 0
        self.bytes = 0
        self.rt_seg_window = rt_seg_window
        self.rt_seg_start = 0
        self.rt_seg_bmr = []
        self.rt_seg_byte_miss = 0
        self.rt_seg_byte_req = 0
        self.result = {}

    def __contains__(self, id):
        return id in self.lru

    def clear(self):
        self.lru.clear()
        self.current_size = 0

    def addToCache(self, id, size):
        x = LRU_Entry(id, size)
        self.lru[id] = x
        self.current_size += size

    def removeFromCache(self, id, size):
        del self.lru[id]
        self.current_size -= size

    def hit(self, id):
        x = self.lru[id]
        self.lru[id] = x

    def evict(self):
        lru = self.lru.popFirst()
        self.current_size -= lru.size
        return lru.id, lru.size

    def miss(self, id, size):
        evicted = []

        if size > self.cache_size:
            print("Too large to fit %d KB object in cache of size %d KB" % (size, self.cache_size))
            return evicted 

        while self.current_size + size > self.cache_size:
            evicted_id, evicted_size = self.evict()
            evicted.append(evicted_id)
        
        self.addToCache(id, size)
        return evicted

    def request(self, id, size):
        miss = True
        evicted = None

        if id in self.lru:
            miss = False
            self.hit(id)
        else:
            evicted = self.miss(id, size)

        op = CacheOp.INSERT if miss else CacheOp.HIT

        return op, evicted

    def run(self):
        progress_bar_size = 30
        progress_bar = ProgressBar(progress_bar_size,
                                   title="{} {}".format(
                                       "LRU", self.cache_size))
        req_count = 0
        for t, id, size in self.reader.read():
            op, evicted = self.request(id, size)
            self.bytes += size
            if op == CacheOp.INSERT:
                self.misses += size
            progress_bar.progress = self.reader.progress
            progress_bar.print()
            req_count += 1
        progress_bar.print_complete()

        return self.misses, self.bytes, [], 0