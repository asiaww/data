import requests
import csv 

with open('negative_list.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(['PACKAGE_NAME', 'SHA256'])

r = requests.get(url="https://api.koodous.com/apks")

while (r.json()['next'] != None):

    for element in r.json()['results']:
        if (element['detected'] == True and element['analyzed'] == True):
            with open('negative_list.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([str(element['package_name']), str(element['sha256'])])

    with open('nexts_list_negative.txt', 'a') as f:
        f.write(str(r.json()['next']))

    r = requests.get(url=r.json()['next'])
