import os
import sys

print("Python executable:", sys.executable)
print("Current Working Directory:", os.getcwd())
print("Files in CWD:", os.listdir("."))
if os.path.exists("/content"):
    print("Files in /content:", os.listdir("/content"))
if os.path.exists("/root"):
    print("Files in /root:", os.listdir("/root"))
