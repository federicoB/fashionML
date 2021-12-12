import json
import os
import urllib.request

import requests
from tqdm.autonotebook import tqdm  # tqdm/tqdm: A Fast, Extensible Progress Bar

os.makedirs("data", exist_ok=True)
# download and parse json containing image URLs
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/zalandoresearch/feidegger/master/data/FEIDEGGER_release_1.1.json")
data = None
with open("FEIDEGGER_release_1.1.json") as f:
    data = json.load(f)

# use requests library Session object to take advantage of HTTP persistent connection
# by reusing TCP connection
s = requests.Session()
for i, element in enumerate(tqdm(data)):
    image = s.get(element['url'])
    with open("data/{}.jpg".format(i), 'wb') as f:
        f.write(image.content)
