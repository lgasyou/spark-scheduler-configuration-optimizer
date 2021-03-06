import os


def file_copy(src_filename, dest_filename):
    with open(src_filename, 'r') as src:
        with open(dest_filename, 'w') as dest:
            dest.write(src.read())


def file_exists(filename: str):
    return os.path.exists(filename) and os.path.isfile(filename)
