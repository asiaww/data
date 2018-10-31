#!/usr/bin/env python
import os, sys
reload(sys)
sys.setdefaultencoding("utf-8")

from xml.dom import minidom
import pandas as pd

def list_into_dict(keys):
    values = [1 for x in keys]
    return dict(zip(keys, values))

def create_index():  
    with open('PERMISSIONS','r') as f:
        index = [line.rstrip('\n') for line in f]

    with open('HARDWARE', 'r') as f:
        index += [line.rstrip('\n') for line in f]

    with open('INTENTS', 'r') as f:
        index += [line.rstrip('\n') for line in f]

    index.append('USES_BROADCAST')
    index.append('LABEL')
    return index

def create_series(data):
    return pd.Series(list_into_dict(data), INDEX, dtype=object).fillna(0)

if __name__ == '__main__':

    INDEX = create_index()
    apps = []
    directories = [x[0] for x in os.walk('.') if 'AndroidManifest.xml' in x[2] and len(x[0]) == 66]

    for element in directories:
        os.chdir('./%s' % str(element))

        RequestedPermissionSet = set()
        BroadcastReceiverSet = set()
        HardwareComponentsSet = set()
        IntentFilterSet = set()

        with open ("AndroidManifest.xml", "r") as f:
            Dom = minidom.parse(f)
            DomCollection = Dom.documentElement

            app = []

            DomPermission = DomCollection.getElementsByTagName("uses-permission")
            for Permission in DomPermission:
                if Permission.hasAttribute("android:name"):
                    RequestedPermissionSet.add(Permission.getAttribute("android:name").encode("utf-8"))
            app += RequestedPermissionSet

            DomBroadcastReceiver = DomCollection.getElementsByTagName("receiver")
            for Receiver in DomBroadcastReceiver:
                if Receiver.hasAttribute("android:name"):
                    app.append('USES_BROADCAST')

            DomHardwareComponent = DomCollection.getElementsByTagName("uses-feature")
            for HardwareComponent in DomHardwareComponent:
                if HardwareComponent.hasAttribute("android:name"):
                    HardwareComponentsSet.add(HardwareComponent.getAttribute("android:name").encode("utf-8"))
            app += HardwareComponentsSet

            DomIntentFilter = DomCollection.getElementsByTagName("intent-filter")
            DomIntentFilterAction = DomCollection.getElementsByTagName("action")
            for Action in DomIntentFilterAction:
                if Action.hasAttribute("android:name"):
                    if 'android.intent.action' in Action.getAttribute("android:name"):
                        IntentFilterSet.add(Action.getAttribute("android:name").encode("utf-8"))
                    else:
                        IntentFilterSet.add('CUSTOM_INTENT')

        if sys.argv[1] == './N/':
            app.append('LABEL')

        apps.append(app)  
        os.chdir('..')

    df = pd.DataFrame([create_series(app) for app in apps], columns=INDEX)
    df.to_csv('./negatywne.csv')
