import importlib

__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'

def found_package(package_name):
    pkg_check = importlib.util.find_spec(package_name)
    found = pkg_check is not None
    return found

