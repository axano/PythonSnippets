import sys


def checkArgs():
    if len(sys.argv) > 1:
        doSmth()
    else:
        usage()
