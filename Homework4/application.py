import sys
from threading import Thread, current_thread
from multiprocessing import Process, Pipe
from time import sleep
from datetime import datetime

def log(message):
    now = datetime.now()
    print(f"[{now.minute}:{now.second}] {message}", file=sys.stderr)

def B(conn, back):
    import codecs
    while True:
        message = conn.recv()
        log(f"(B) Received: '{message}'")
        if message == "CANCEL": break
        encoded = codecs.encode(message, "rot_13")
        back.send(encoded)

def A(conn):
    parent, child = Pipe()
    b = Process(target=B, args=(child, conn))
    b.start()
    while True:
        sleep(5)
        message = conn.recv()
        log(f"(A) Received: '{message}'")
        if message == "CANCEL":
            parent.send(message)
            break
        parent.send(message.lower())
    b.join()

def main():
    parent, child = Pipe(duplex=True)
    a = Process(target=A, args=(child,))
    a.start()

    def receiver():
        t = current_thread()
        while not t.cancel:
            message = parent.recv()
            log(f"(Main) Received: '{message}'")

    t = Thread(target=receiver)
    t.cancel = False
    t.start()
    
    while c := input():
        parent.send(c)
        if c == "CANCEL":
            break

    parent.send("CANCEL")
    t.cancel = True
    t.join()
    a.join()
    
if __name__ == "__main__":
    main()
