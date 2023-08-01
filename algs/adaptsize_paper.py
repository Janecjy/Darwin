from bloom_filter2 import BloomFilter
from collections import deque
import getopt
import sys
import copy
import math
import random
import numpy as np

from lru import LRU

WARMUP_LENGTH = 1000000

trace_path = (
    output_dir
) = freq_thres = size_thres = hoc_s = dc_s = collection_length = climb_step = None


def parseInput():
    global trace_path, output_dir, hoc_s, dc_s, collection_length

    # Get the arguments from the command-line except the filename
    argv = sys.argv[1:]

    try:
        # Define the getopt parameters
        opts, args = getopt.getopt(
            argv,
            "t:o:h:d:l:",
            ["trace_path", "output_dir", "hoc_size", "dc_size", "collection_length"],
        )
        # Check if the options' length is 3
        if len(opts) != 5:
            print(
                "usage: adaptsize.py -t <trace_path> -o <output_dir> -h <HOC_size> -d <DC_size> -l <collection_length>"
            )
        else:
            # Iterate the options and get the corresponding values
            for opt, arg in opts:
                if opt == "-t":
                    trace_path = arg
                if opt == "-o":
                    output_dir = arg
                if opt == "-h":
                    hoc_s = int(arg)
                if opt == "-d":
                    dc_s = int(arg)
                if opt == "-l":
                    collection_length = int(arg)

    except getopt.GetoptError as err:
        # Print help message
        print(err)
        print(
            "usage: adaptsize.py -t <trace_path> -o <output_dir> -h <HOC_size> -d <DC_size> -l <collection_length>"
        )
        sys.exit(2)


# p_i_in_cache(r_i, s_i, c, u_c) defined by Thm 1 in the paper
def p_i_in_cache(r_i, s_i, c, u_c):
    prod = np.exp((-c * s_i) + (r_i / u_c)) - np.exp(-c * s_i)
    return (prod) / (1 + prod)

# the expected size given r, s, c, and u_c is the sum of (s_i * p_i_in_cache)
def expected_total_size(r, s, c, u_c):
    return np.dot(s, p_i_in_cache(r, s, c, u_c))


# use bisection for u_c in (0, 1) to find the value that leads to the expected size of cache to equal HOC_SIZE
def search_u_c(r, s, c, target_size):
    L = 0.0
    R = 1.0
    u_c = 0
    # bisection
    for _ in range(40):
        u_c = (L + R) / 2
        result = expected_total_size(r, s, c, u_c)
        if result < target_size:
            R = u_c
        else:
            L = u_c
    return u_c


# predict the OHR for a given c parameter, given object request rates and sizes, and total HOC size
def measure_ohr_c(r, s, c, HOC_SIZE):
    # find the u_c for this c
    found_u_c = search_u_c(r, s, c, HOC_SIZE)
    if found_u_c == np.nan:
        return 0
    expected_size = expected_total_size(r, s, c, found_u_c)
    # if the expected size is not valid or not within 1% of HOC size we have a weird c or u_c, disregard it
    if expected_size == np.nan or ((expected_size - HOC_SIZE) / HOC_SIZE > 0.01):
        return 0

    # the hit rate is the sum of (r_i * P_i_in_cache)
    OHR = np.dot(r, p_i_in_cache(r, s, c, found_u_c))

    return OHR


# given r (request frequency) and s (size) vectors for the objects requested in the last delta requests, find the c value that maximizes predicted OHR
def pick_best_c(r, s, HOC_SIZE):
    # make a vector of potential c values from 2^-2 to 2^-20 jumping by 0.25 in the exponent
    c_s = np.reciprocal(np.power(2, np.arange(2, 20, 0.25)))
    # make a vectorized function to measure ohr for a given c
    vec_func = np.vectorize(measure_ohr_c, excluded=set([0, 1, 3]))
    # calculate the hit rate for each c value
    hit_rates = vec_func(r, s, c_s, HOC_SIZE)
    # if we could not calculate the hit rate, replace with zero
    np.nan_to_num(hit_rates, copy=False)
    # return the c with the best hit rate
    return c_s[np.argmax(hit_rates)]


class Cache:
    def __init__(self, hoc_s, dc_s, c, delta):
        self.dc = LRU(dc_s, {})
        self.hoc = LRU(hoc_s, {})
        self.hoc_size = hoc_s
        self.c = c
        self.requestLog = deque()  # store the last delta requests [id, id, id, ...]
        self.sizeTab = dict()  # store the size of objects we have seen {id: size}
        self.bloom = BloomFilter(max_elements=1000000, error_rate=0.1)

        self.delta = delta

        # record period of stats for this cache
        self.obj_hit = 0
        # self.byte_miss = 0
        self.disk_write = 0
        self.debug = False

    def copy_state(self, cache):
        # fix this by using iteration to do deepcopy
        self.dc.copy_state(cache.dc)
        self.hoc.copy_state(cache.hoc)
        self.c = cache.c
        self.requestLog = copy.deepcopy(cache.requestLog)
        self.sizeTab = copy.deepcopy(cache.sizeTab)

    def reset(self, new_c):
        print("new c:", new_c)
        self.c = new_c
        self.obj_hit = self.disk_write = 0

    # given the past delta requests, find r_i, s_i, then find the best c value
    def recalculate_c(self):
        # store r in a vector
        r_counter = dict()
        r = []
        s = []
        # count occurrences of the ids
        for id in self.requestLog:
            r_counter[id] = r_counter.get(id, 0) + 1

        for id in r_counter:
            assert id in self.sizeTab, "don't know size of object %d" % id

            r.append(r_counter[id] / self.delta)
            s.append(self.sizeTab[id])
        # now we have the request rate and size of each object that has been requested in the past delta requests
        r = np.array(r)
        s = np.array(s)
        # construct an OHR vs c curve, and pick the best c
        return np.reciprocal(pick_best_c(r, s, self.hoc_size))

    # follow the formula for admission from the paper
    def determine_admission(self, size):
        # calculate e^(-size/c) and if it is less than rand(0,1) then admit
        return math.exp(-size / self.c) < random.random()

    def request(self, t, id, size):
        self.requestLog.append(id)
        # store only the last delta requests
        if len(self.requestLog) > self.delta:
            self.requestLog.popleft()
        obj_hit = 0
        disk_write = 0

        global tot_num

        self.sizeTab[id] = size

        if id in self.hoc:
            self.hoc.hit(id)
            obj_hit = 1
            # print("Object hit: ", tot_num, id, obj_hit)
        elif id in self.dc:
            self.dc.hit(id)
            # random admission occurs here, bring it into HOC
            if self.determine_admission(size):
                if self.debug:
                    print("call promote")
                self.promote(id, size)
            if tot_num >= WARMUP_LENGTH:
                disk_write += size / 4
        else:
            if self.debug:
                print("call miss")
            evicted = self.dc.miss(id, size)

            if tot_num >= WARMUP_LENGTH:
                disk_write += size / 4
        self.bloom.add(id)

        if tot_num >= WARMUP_LENGTH:
            self.obj_hit += obj_hit
            self.disk_write += disk_write

        return obj_hit, disk_write

    # promote an object from dc to hoc
    def promote(self, id, size):
        # delete the object from dc and the dc access table
        self.dc.removeFromCache(id, size)

        # when hoc is full, demote the hoc's lru object to the dc
        while self.hoc.current_size + size > self.hoc.cache_size:
            self.demote()

        # add the object to hoc
        self.hoc.addToCache(id, size)

    # demote the lru object from hoc to dc
    def demote(self):
        global isWarmup
        isWarmup = False

        # evict the lru object from hoc
        id, size = self.hoc.evict()

        # add the object to dc

    # find the number of times the object with this id has been requested in the last delta requests
    def countFreq(self, id):
        return self.requestLog.count(id)


def run():
    global currentT, disk_write, collection_length

    real_cache = Cache(
        hoc_s, dc_s, 64, collection_length
    )  # TODO: find default c value, arbitrary value right now
    real_cache.debug = False
    global tot_num
    tot_num = (
        tot_req
    ) = tot_bytes = tot_obj_hit = tot_byte_miss = tot_hoc_hit = tot_disk_write = 0

    global isWarmup
    isWarmup = True

    with open(trace_path) as fp:
        for line in fp:
            try:
                line = line.split(",")
                t = int(line[0])
                id = int(line[1])
                size = int(line[2])
                currentT = t
            except:
                print(trace_path, line, file=sys.stderr)
                continue

            obj_hit, disk_write = real_cache.request(t, id, size)

            if tot_num >= WARMUP_LENGTH:
                if obj_hit == 1:
                    tot_hoc_hit += 1
                tot_obj_hit += obj_hit
                tot_disk_write += disk_write
                tot_req += 1
                tot_bytes += size
            if tot_num > WARMUP_LENGTH and tot_num % collection_length == 0:
                print(tot_num)
                print(
                    "real cache stage hoc hit: {:.4f}%, disk write: {:.4f}".format(
                        real_cache.obj_hit / collection_length * 100,
                        real_cache.disk_write,
                    )
                )
                print(
                    "hoc hit: {:.4f}%, hr: {:.4f}%, bmr: {:.4f}%, disk write: {:.4f}".format(
                        tot_hoc_hit / tot_req * 100,
                        tot_obj_hit / tot_req * 100,
                        tot_byte_miss / tot_bytes * 100,
                        tot_disk_write,
                    )
                )
                sys.stdout.flush()
                # solve for new c value and replace it as the new parameter
                real_cache.reset(real_cache.recalculate_c())

                sys.stdout.flush()
            tot_num += 1

    print(
        "real cache stage hoc hit: {:.4f}%, disk write: {:.4f}".format(
            real_cache.obj_hit / collection_length * 100, real_cache.disk_write
        )
    )
    print(
        "final hoc hit: {:.4f}%, hr: {:.4f}%, bmr: {:.4f}%, disk write: {:.4f}".format(
            tot_hoc_hit / tot_req * 100,
            tot_obj_hit / tot_req * 100,
            tot_byte_miss / tot_bytes * 100,
            tot_disk_write,
        )
    )
    sys.stdout.flush()


def main():
    parseInput()
    if None not in (trace_path, hoc_s, dc_s, collection_length):
        print(
            "trace: {}, HOC size: {}, DC size: {}, collection length: {}".format(
                trace_path, hoc_s, dc_s, collection_length
            )
        )
    else:
        sys.exit(2)

    run()

    return 0


if __name__ == "__main__":
    main()
