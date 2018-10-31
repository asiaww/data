import os
import sys
import subprocess

walk_dir = sys.argv[1]
#-------------------------------------decompile----------------------------

i = 0

apktool = "/bin/apktool.jar"

for root, subdirs, files in os.walk(walk_dir):
 list_file_path = os.path.join(root, 'my-directory-list.txt') 
 
 with open(list_file_path, 'wb') as list_file:
    for file in files:
        if file.endswith(".apk"):
            i=i+1
            print str(i)+"-"+file 
            os.system("java -jar " + apktool + " d " + root+"/"+file)
