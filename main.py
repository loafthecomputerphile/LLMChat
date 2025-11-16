import requests, pprint

response = requests.get("https://api.github.com/repos/jgm/pandoc/releases/latest")
pprint.pprint(response.json())