import concurrent.futures
import requests
import threading
import time
import math
import toolz
import typing
import sys

thread_local = threading.local()

class Timer():
    """
    Basic timer class for measuring runtime.
    """
    def __init__(self, state : str = "idle"):
        self.elapsedTime = 0e0
        self.startTime = 0e0
        self.stopTime = 0e0
        if state != "idle":
            self.state = "running"
        else:
            self.state = "idle"
    def start(self,) -> float:
        if self.state == "idle":
            self.state = "running"
            self.startTime = time.perf_counter()
            self.elapsedTime = 0
        else:
            self.elapsedTime = time.perf_counter() - self.startTime
        return self.elapsedTime
    def restart(self,) -> float:
        self.state = "running"
        self.startTime = time.perf_counter()
        self.elapsedTime = 0
        return self.elapsedTime
    def stop(self,) -> float:
        if self.state == "running":
            self.state = "idle"
            self.stopTime = time.perf_counter()
            self.elapsedTime = self.stopTime - self.startTime
        else:
            self.state = "idle"
        return self.elapsedTime
    def getElapsed(self,) -> float:
        if self.state == "running":
            self.elapsedTime = time.perf_counter() - self.startTime
        return self.elapsedTime

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


if __name__ == "__main__":
    sites = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 80
    start_time = time.time()
    download_all_sites(sites)
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} in {duration} seconds")

def testRun():
    sites = [
        "https://www.jython.org",
        "http://olympus.realpython.org/dice",
    ] * 80
    start_time = time.time()
    download_all_sites(sites)
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} in {duration} seconds")


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
isPrime(7)
isPrime(8)


timer.start()
l = [i for i in range(10000000) if isPrime(i)]
print(timer.stop())

executor = concurrent.futures.ThreadPoolExecutor(max_workers=5,)

l = executor.map(isPrime, list(range(1000)))
l = list(l)

l = [i for i in range(len(l)) if l[i]==True]


l = executor.map(math.sqrt, range(1000))

toolz.reduce(lambda x,y: x+y, range(100))

toolz.reduce(lambda x,y : x and y, (False for i in range(1000000000)), True)
