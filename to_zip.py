#!/usr/bin/env python
import os, sys
import zipfile
import shutil

if __name__ == '__main__':
    directories = [x[0] for x in os.walk('.') if 'AndroidManifest.xml' in x[2]]

    for element in directories:
        shutil.make_archive(element, 'zip', element)        

    #archives = [x[2] for x in os.walk(sys.argv[1])]
    #for lists in archives:
    #    for element in lists:
    #        base = os.path.splitext(element)[0]
    #        os.rename(sys.argv[1] + element, base + ".apk")

