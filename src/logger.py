import logging
import os
import sys

file_name = os.path.join("logs", "test.log")
file_dir = os.path.dirname(file_name)
os.makedirs(file_dir, exist_ok=True)
curr_file = os.path.basename(sys.argv[0])

logging.basicConfig(
    filename=file_name,
    format="%(asctime)s %(filename)s %(levelname)s  %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    level=logging.DEBUG,
    filemode="w"
)
