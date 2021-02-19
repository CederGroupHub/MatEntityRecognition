import json
from pprint import pprint
import pymongo
import os


__author__ = 'Tanjin He'
__maintainer__ = 'Tanjin He'
__email__ = 'tanjin_he@berkeley.edu'


###########################################
# communicate with mongodb
###########################################

def get_mongo_db(config_file_path=None, mongo_config=None):
    """
    read config file and return mongo db object

    :param config_file_path: (str or None)
        If it is str, which is path to the config.json file. config is a dict of dict as
            config = {
                'mongo_db': {
                    'host': 'mongodb05.nersc.gov',
                    'port': 27017,
                    'db_name': 'COVID-19-text-mining',
                    'username': '',
                    'password': '',
                }
            }
    :return: (obj) mongo database object
    """
    if config_file_path and os.path.exists(config_file_path):
        with open(config_file_path, 'r') as fr:
            config = json.load(fr)

        client = pymongo.MongoClient(
            host=config['mongo_db']['host'],
            port=config['mongo_db']['port'],
            username=config['mongo_db']['username'],
            password=config['mongo_db']['password'],
            authSource=config['mongo_db']['auth_source'],
        )

        db = client[config['mongo_db']['db_name']]
    elif mongo_config:
        client = pymongo.MongoClient(
            host=mongo_config['host'],
            port=mongo_config['port'],
            username=mongo_config['username'],
            password=mongo_config['password'],
            authSource=mongo_config['auth_source'],
        )
        db = client[mongo_config['db_name']]
    else:
        client = pymongo.MongoClient(
            host=os.getenv("MONGO_HOST"),
            port=os.getenv("MONGO_PORT"),
            username=os.getenv("MONGO_USER"),
            password=os.getenv("MONGO_PASS"),
            authSource=os.getenv("MONGO_AUTH_SOURCE")
        )
        db = client[os.getenv("MONGO_DB")]
    return db

if __name__ == '__main__':
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../config/config.json',
    )
    print('config_path', config_path)
    db = get_mongo_db(config_path)
    print(db.collection_names())
