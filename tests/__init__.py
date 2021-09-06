
import os
import sys


PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH,"clim_snr"
)
sys.path.append(SOURCE_PATH)

if __name__ == '__main__':
    print(PROJECT_PATH)
    print(SOURCE_PATH)