import gzip
import os
import shutil

import tarfile


def extract_tar_file(save_dir, tar_filename):
    tar = tarfile.open(os.path.join(save_dir, tar_filename), 'r:gz')
    tar.extractall(path=save_dir)
    tar.close()


def extract_gzip_file(save_dir, gzip_filename, filename):
    with gzip.open(os.path.join(save_dir, gzip_filename), 'rb') as f_in:
        with open(os.path.join(save_dir, filename), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
