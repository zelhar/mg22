import concurrent.futures
import requests
import threading
import time
import math
import toolz
import typing
import sys
from numba import jit

thread_local = threading.local()

class Timer():
    """
    Basic timer class for measuring runtime.
    """
    def __init__(self, state : str = "idle"):
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
    def getState(self,) -> str:
        return self._state
    def start(self,) -> float:
        """
        if was idle start counting. otherwise just measure current time lapse.
        return time lapsed (counter)
        """
        if self._state == "idle": # start counting from 0
            self._state = "running"
            self._previous = time.perf_counter()
            self.counter = 0
        else: # had already been running
            self.counter = time.perf_counter() - self._previous
        return self.counter
    def restart(self,) -> float:
        """
        always reset counter and initiate new counting.
        """
        self._state = "running"
        self._previous= time.perf_counter()
        self.counter = 0
        return self.counter
    def stop(self,) -> float:
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
    def getCount(self,) -> float:
        """
        returns time counter without changing _state
        """
        if self._state == "running":
            self.counter = time.perf_counter() - self._previous
        return self.counter


def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


def download_site(url):
    session = get_session()
    with session.get(url) as response:
        print(f"Read {len(response.content)} from {url}")


def download_all_sites(sites):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download_site, sites)


def testRun():
    sites = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 80
    start_time = time.time()
    download_all_sites(sites)
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} in {duration} seconds")


#@jit(nopython=True,)
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
        n = 1 + math.sqrt(y).__int__()
        for i in range(2,n):
            if y % i == 0:
                return False
    return True

def primeTest(n=10000000, threads=8, chunksize=1000,):
    timer = Timer()
    print("starting single threaded task")
    timer.start()
    l = [isPrime(i) for i in range(n) ]
    timer.stop()
    print("took it {} time".format(timer.getCount()))
    # for io speedup use ThreadPoolExecutor and for cpu speedup ProcessPoolExecutor
    # probably because of GIL, threading is very slow for cpu bound tasks
    # chunksize only used for process, and should be sufficently large as long
    # as free memory is available.
    print("now multithread")
    #executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads,)
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=threads,)
    timer.start()
    l = executor.map(isPrime, range(n), chunksize=chunksize)
    #print(list(toolz.take(10, l)))
    timer.stop()
    print("took it {} time".format(timer.getCount()))

def foo(x,y,z):
    return x+y+z
f = toolz.partial(foo, y=1,z=2)
mylist = [i for i in range(10)]
nums = list(map(f, mylist))
executor = concurrent.futures.ProcessPoolExecutor(max_workers=4,)
nums = list(executor.map(f, mylist))

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

