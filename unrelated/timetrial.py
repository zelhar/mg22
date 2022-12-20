import time
import sys
import typing as T
import toolz

class Tester:
    """
    a wrapper class meant to be given
    a callabel f and a tuple params,
    will start time counting and then execute
    f(*params)
    outputs the elapsed time.
    """

    def __init__(
        self,
        #f: T.Callable,
        #params: T.Tuple,
        expr : str,
    ) -> None:
        #self.f = f
        #self.params = params
        self.expr = expr
        return
    def run(self,) -> float:
        start = time.perf_counter()
        #self.f(*self.params)
        eval(self.expr)
        stop = time.perf_counter()
        elapsed_time = stop - start
        return elapsed_time



if __name__ == "__main__":
    #f = toolz.curried.reduce(lambda x, y : x+y,)
    if len(sys.argv) <= 1:
        expr = """(
toolz.curried.reduce(lambda x, y : x+y,
    range(100000),)) """
    else:
        expr = "".join(sys.argv[1:])
    test = Tester(expr)
    t = test.run()
    print(t)
    print("argv = ", sys.argv)
    print("expr = ", expr)


