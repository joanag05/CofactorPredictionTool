from zipfile import ZipFile
import os

def read_conf_file(path):
    '''
    reading the configuration file
    It must be assembled in the following manner: key = configuration value
    It will return a dictionary in the following format: {workers : 3, limit : 100}
    '''
    res = {}

    with open(path) as file:
        lines = file.readlines()
        for line in lines:
            if '=' in line:
                prop = line.replace(" ", "").replace("\n", "")
                l_str = prop.split('=')
                res[l_str[0]] = l_str[1]

    return res


def read_workers_conf_file(path):
    res = []

    with open(path) as file:
        lines = file.readlines()
        for line in lines:
            if "#" not in line[0]:
                str = line.replace(" ", "").replace("\n", "")
                res.append(str)

    return res

def compressFiles(path, savePath):
    file_paths = get_all_file_paths(path)

    zipLocation = savePath

    with ZipFile(zipLocation, 'w') as zip:
        # writing each file one by one
        for file in file_paths:
            new_file = file.split("/")[-1]
            zip.write(file, arcname=new_file)

    return zipLocation

def get_all_file_paths(directory):

    # initializing empty file paths list
    file_paths = []

    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):

        for filename in files:

            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    # returning all file paths