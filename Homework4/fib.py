from threading import Thread
from multiprocessing import Process, Pipe
from timeit import default_timer as timer

bigN = 35

def fib(n: int) -> int:
    if n < 2: return n
    return fib(n - 1) + fib(n - 2)

class FibThread(Thread):
    maxThreads = 10
    threadsUsed = 0
    def __init__(self, n: int):
        super().__init__()
        FibThread.threadsUsed += 1
        self.n = n
        self._result = None
        self.start()
    
    def run(self):
        if self.n < 2: 
            self._result = self.n
            return

        assert FibThread.threadsUsed <= FibThread.maxThreads
        if FibThread.threadsUsed == FibThread.maxThreads:
            r1 = fib(self.n - 1)
            r2 = fib(self.n - 2)
            self._result = r1 + r2
        else:
            r1 = FibThread(self.n - 1)
            r2 = fib(self.n - 2)
            self._result = r1.result() + r2

    def result(self) -> int:
        self.join()
        return self._result

class FibProcess(Process):
    maxProcesses = 10
    def __init__(self, n: int, conn, used: int = 0):
        super().__init__()
        self.used = used + 1 # count ourselves
        self.n = n
        self.conn = conn
    
    def run(self):
        if self.n < 2: 
            self.conn.send(self.n)
            return

        assert self.used <= FibProcess.maxProcesses
        if self.used == FibProcess.maxProcesses:
            r1 = fib(self.n - 1)
            r2 = fib(self.n - 2)
            self.conn.send(r1 + r2)
        else:
            getEnd, sendEnd = Pipe()
            r1 = FibProcess(self.n - 1, sendEnd, self.used)
            r1.start()
            r2 = fib(self.n - 2)
            result1 = int(getEnd.recv())
            r1.join()
            self.conn.send(result1 + r2)

if __name__ == "__main__":
    correct = fib(bigN)

    avg = 0
    for i in range(10):
        start = timer()
        c = fib(bigN)
        assert c == correct
        end = timer()
        avg += end - start
    print("Sequenced: ", avg / 10, "seconds")

    avg = 0
    for i in range(10):
        start = timer()
        t = FibThread(bigN)
        c = t.result()
        assert c == correct
        end = timer()
        FibThread.threadsUsed = 0
        avg += end - start
    print("Threaded: ", avg / 10, "seconds")

    avg = 0
    for i in range(10):
        start = timer()
        getEnd, sendEnd = Pipe()
        t = FibProcess(bigN, sendEnd)
        t.start()
        c = getEnd.recv()
        t.join()
        assert c == correct
        end = timer()
        FibProcess.processesUsed = 0
        avg += end - start
    print("Multi-processed: ", avg / 10, "seconds")
