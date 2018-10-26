import sys
from xml.dom import minidom

def extractFeaturesFromManifest(manifestFile):

    permissions = set()
    services = set()
    broadcastReceivers = set()
    hardwareComponents = set()
    intentFilters = set()

    with open(manifestFile, "r") as f:
        
        dom = minidom.parse(f)
        domCollection = Dom.documentElement

        domPermission = DomCollection.getElementsByTagName("uses-permission")
        for permission in domPermission:
            if permission.hasAttribute("android:name"):
                permissions.add(permission.getAttribute("android:name"))

        domService = domCollection.getElementsByTagName("service")
        for service in domService:
            if service.hasAttribute("android:name"):
                services.add(service.getAttribute("android:name"))

        domBroadcastReceiver = domCollection.getElementsByTagName("receiver")
        for receiver in domBroadcastReceiver:
            if receiver.hasAttribute("android:name"):
                broadcastReceivers.add(receiver.getAttribute("android:name"))

        domHardwareComponent = domCollection.getElementsByTagName("uses-feature")
        for hardwareComponent in domHardwareComponent:
            if hardwareComponent.hasAttribute("android:name"):
                hardwareComponents.add(hardwareComponent.getAttribute("android:name"))

        domIntentFilter = domCollection.getElementsByTagName("intent-filter")
        domIntentFilterAction = domCollection.getElementsByTagName("action")
        for action in domIntentFilterAction:
            if action.hasAttribute("android:name"):
                intentFilters.add(action.getAttribute("android:name"))

    return permissions, services, broadcastReceivers, hardwareComponents, intentFilters

print(extractFeaturesFromManifest(sys.argv[1])
