import socket

#Does NOT need routable net access or any connection at all.
#Works even if all interfaces are unplugged from the network.
#Does NOT need or even try to get anywhere else.
#Works with NAT, public, private, external, and internal IP's
#Pure Python 2( or 3) with no external dependencies.
#Works on Linux, Windows, and OSX.

def getLocalIp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP