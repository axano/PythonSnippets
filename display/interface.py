import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    RED = '\033[31m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


#This will work on unixes including OS X, linux and windows
# (provided you use ANSICON, or in Windows 10 provided you enable VT100 emulation).
def printWarning(message):
    print(bcolors.WARNING+"[ "+str(time.ctime())+" ] WARNING: "+message+bcolors.ENDC)

def printError(message):
    print(bcolors.ERROR+"[ "+str(time.ctime())+" ] ERROR: "+message+bcolors.ENDC)
    
def printSuccess(message):
    print(bcolors.OKGREEN+"[ "+str(time.ctime())+" ] SUCCESS: "+message+bcolors.ENDC)
