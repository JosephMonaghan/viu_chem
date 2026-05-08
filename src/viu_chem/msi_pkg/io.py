import os
import re


def files_match_pattern(file_dir, pattern):
    """Returns files in a directory whose names match a regex pattern.
    
    :param file_dir: Directory to search
    :param pattern: Regex pattern to match filenames against
    :return: List of matching filenames"""
    file_list = []
    for f in os.listdir(file_dir):
        if re.search(pattern,f):
            file_list.append(f)
    return file_list


def decode_files(file_list):
    """Extracts class, group, and sample codes from underscore-delimited filenames.
    
    :param file_list: List of filenames to decode
    :return: Tuple of class, group, and sample code lists"""
    classes = set()
    groups = set()
    samples = set()
    for f in file_list:
        f = f.split('.')[0]
        classes.add(f.split('_')[0])
        groups.add(f.split('_')[1])
        samples.add(f.split('_')[2])
    return list(classes), list(groups), list(samples)
