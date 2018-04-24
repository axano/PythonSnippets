import socket
import fcntl
import struct


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



#needs an network interface card name as parameter
#returns the netmask in n.n.n.n form
#tested in python 3.6
#Tested on UNIX-like OS's
def getNetmaskByNic(ifname):
    return socket.inet_ntoa(fcntl.ioctl(socket.socket(socket.AF_INET, socket.SOCK_DGRAM), 35099, struct.pack(bytes('256s','utf-8'), bytes(ifname,'utf-8')))[20:24])

#Searches the iface that is connected to the gateway
#by looking it up in /proc/net/route
#Tested in python 3.6
#Tested on UNIX-like OS's
def getDefaultIfaceNameLinux():
    route = "/proc/net/route"
    with open(route) as f:
        for line in f.readlines():
            try:
                iface, dest, _, flags, _, _, _, _, _, _, _, =  line.strip().split()
                if dest != '00000000' or not int(flags, 16) & 2:
                    continue
                return iface
            except:
                continue
