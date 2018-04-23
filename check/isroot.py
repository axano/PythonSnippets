import os

if os.getuid() == 0:
    print("user is root")
else:
    print("Please run program as root.")
    print("Exiting...")
