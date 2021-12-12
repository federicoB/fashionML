import json
import os
import urllib.request

import requests
from docopt import docopt
from tqdm.autonotebook import tqdm  # tqdm/tqdm: A Fast, Extensible Progress Bar

feidegger_file_name = "FEIDEGGER_release_1.1.json"
feidegger_url = "https://raw.githubusercontent.com/zalandoresearch/feidegger/master/data/FEIDEGGER_release_1.1.json"
data_folder = "data/"


def download_dataset(percent=100):
    if not os.path.exists(feidegger_file_name):
        # download and parse json containing image URLs
        urllib.request.urlretrieve(feidegger_url, feidegger_file_name)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        data = None
        with open(feidegger_file_name) as f:
            data = json.load(f)
        # use requests library Session object to take advantage of HTTP persistent connection
        # by reusing TCP connection
        s = requests.Session()
        total = int((len(data) / 100) * percent)
        for i, element in enumerate(tqdm(data[:total])):
            image = s.get(element['url'])
            with open(data_folder + "{}.jpg".format(i), 'wb') as f:
                f.write(image.content)


if __name__ == '__main__':
    # usage pattern
    usage = '''
     
    Usage:
      download_dataset.py
      download_dataset.py --percent <argument>
     
    Options:
      -p, --percent     Display help
           
    '''

    args = docopt(usage)
    download_dataset(args['--percent'])
