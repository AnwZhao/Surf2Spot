import os
import shutil

def path_exists(path):
    if not os.path.exists(path):
        raise ValueError(f"{path} does not exist.")
        return False
    else:
        return True

def mk_dir(file_path):
    os.makedirs(file_path, exist_ok=True)
