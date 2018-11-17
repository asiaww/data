import text_to_image
import pandas as pd
import os
import operator
from PIL import Image

def list_into_dict(keys):
    dictionary = {}
    dictionary['label'] = str(keys[0])
    
    for i in range(0,len(keys)-1):
        dictionary['pixel{0}'.format(i)] = str(keys[i+1])

    return dictionary

def create_index():
    index = []
    index.append('label')    
    index += ['pixel{0}'.format(i) for i in range(0,784)]
    return index

def create_series(dataa):
    return pd.Series(list_into_dict(dataa), INDEX, dtype=object)

df = pd.read_csv('DATA_FULL.csv')

#get header as list
header = list(df.columns.values)

cmap = {'0': (255,255,255),
        '1': (0,0,0)}

#for each row get values, if value equals 1 get column name for it and concatenate it into long string

INDEX = create_index()
apps = []

for row in df.iterrows():
    app = []
    app.append(row[1]['LABEL'])

    string = ''
    values = row[1].values.tolist()
    for index in range(0,len(values)):
        if values[index] == 1:
            string += '{0}\n'.format(header[index])

    bit_encoding = ''.join(format(ord(x), 'b') for x in string)
    data = [cmap[letter] for letter in bit_encoding if letter != ' ']
    img = Image.new('RGB', (100,100), "white")
    try:
        img.putdata(data)
    except TypeError:
        print(row[0])
        apps.append([255 for _ in range(0,784)])
                
    img = img.resize((28,28),Image.ANTIALIAS)
    img.save("{0}.png".format(row[0]))
    for pixel in list(img.getdata()):
        app.append(pixel[0])

    apps.append(app)
    i += 1

df2 = pd.DataFrame([create_series(app) for app in apps], columns=INDEX)
df2.to_csv('./obrazy_out.csv')
