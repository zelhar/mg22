import concurrent.futures
import requests
import threading
import time
import math
import toolz
import functools
import itertools
import operator
import typing
import sys
from numba import jit, njit

class Timer:
    """
    Basic timer (AKA stopwatch) class for measuring runtime.
    """

    def __init__(self, state: str = "idle"):
        """
        if state != 'idle'  it will
        be counting from its initiation.
        default is idle.
        """
        self.counter = 0e0
        self._previous = -9e10
        self._state = "idle"
        if state != "idle":
            self._previous = time.perf_counter()
            self._state = "running"
        else:
            self._state = "idle"
        return

    def getState(
        self,
    ) -> str:
        return self._state

    def start(
        self,
    ) -> float:
        """
        if was idle start counting. otherwise just measure current time lapse.
        return time lapsed (counter)
        """
        if self._state == "idle":  # start counting from 0
            self._state = "running"
            self._previous = time.perf_counter()
            self.counter = 0
        else:  # had already been running
            self.counter = time.perf_counter() - self._previous
        return self.counter

    def restart(
        self,
    ) -> float:
        """
        always reset counter and initiate new counting.
        """
        self._state = "running"
        self._previous = time.perf_counter()
        self.counter = 0
        return self.counter

    def stop(
        self,
    ) -> float:
        """
        stop running if it were running.
        returns counter.
        """
        if self._state == "running":
            self.counter = time.perf_counter() - self._previous
            self._state = "idle"
        else:
            self._state = "idle"
        return self.counter

    def getCount(
        self,
    ) -> float:
        """
        returns time counter without changing _state
        """
        if self._state == "running":
            self.counter = time.perf_counter() - self._previous
        return self.counter

@jit(nopython=True, nogil=True,)
def isPrime(x : typing.Union[int,float]) -> bool:
    """
    returns True iff int(x) is prime
    """
    y = int(abs(x))
    if y < 1:
        return False
    elif y < 4:
        return True
    else:
        n = int(1 + math.sqrt(y))
        for i in range(2,n):
            if y % i == 0:
                return False
    return True

def isPrimeNoJit(x : typing.Union[int,float]) -> bool:
    """
    returns True iff int(x) is prime
    """
    y = int(abs(x))
    if y < 1:
        return False
    elif y < 4:
        return True
    else:
        n = int(1 + math.sqrt(y))
        for i in range(2,n):
            if y % i == 0:
                return False
    return True


def primeTest(n=10000000, threads=8, chunksize=1000, jit=True):
    timer = Timer()
    print("starting single threaded task\n")
    print("first without jit:")
    timer.start()
    l = (isPrimeNoJit(i) for i in range(n) )
    #l = list(map(isPrimeNoJit, range(n)))
    timer.stop()
    print("took it {} time\n".format(timer.getCount()))
    print("now with jit:")
    timer.start()
    l = [isPrime(i) for i in range(n) ]
    timer.stop()
    print("took it {} time\n".format(timer.getCount()))
    # for io speedup use ThreadPoolExecutor and for cpu speedup ProcessPoolExecutor
    # probably because of GIL, threading is very slow for cpu bound tasks
    # chunksize only used for process, and should be sufficently large as long
    # as free memory is available.
    print("now multithread\n")
    #executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads,)
    print("first without jit:")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=threads,)
    timer.start()
    l = executor.map(isPrimeNoJit, range(n), chunksize=chunksize)
    l = list(l)
    #print(list(toolz.take(10, l)))
    timer.stop()
    print("took it {} time\n".format(timer.getCount()))
    print("now with jit:")
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=threads,)
    timer.start()
    l = executor.map(isPrime, range(n), chunksize=chunksize)
    l = list(l)
    timer.stop()
    print("took it {} time\n".format(timer.getCount()))
    l = list(toolz.take(30, l))
    print([i for i in range(len(l)) if l[i]==True],)
    

if __name__ == "__main__":
    n = 1000000
    chunksize = 10000
    threads = 8
    if len(sys.argv) > 1:
        n=int(sys.argv[1])
    if len(sys.argv) > 2:
        chunksize=int(sys.argv[2])
    if len(sys.argv) > 3:
        threads = int(sys.argv[3])
    primeTest(n, chunksize=chunksize, threads=threads)

