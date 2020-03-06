import json
import hashlib
import datetime
import copy
import cv2
import threading
import numpy as np
from collections import Counter, defaultdict
import itertools
import pyarrow.plasma as plasma
import SharedArray as sa
import matplotlib.pyplot as plt
from frigate.util import draw_box_with_label
from frigate.edgetpu import load_labels
import requests
import pathlib

PATH_TO_LABELS = '/labelmap.txt'

LABELS = load_labels(PATH_TO_LABELS)
cmap = plt.cm.get_cmap('tab10', len(LABELS.keys()))

COLOR_MAP = {}
for key, val in LABELS.items():
    COLOR_MAP[val] = tuple(int(round(255 * c)) for c in cmap(key)[:3])

class TrackedObjectProcessor(threading.Thread):
    def __init__(self, config, client, topic_prefix, tracked_objects_queue):
        threading.Thread.__init__(self)
        self.config = config
        self.client = client
        self.topic_prefix = topic_prefix
        self.tracked_objects_queue = tracked_objects_queue
        self.plasma_client = plasma.connect("/tmp/plasma")
        self.camera_data = defaultdict(lambda: {
            'best_objects': {},
            'object_status': defaultdict(lambda: defaultdict(lambda: 'OFF')),
            'tracked_objects': {},
            'current_frame': np.zeros((720,1280,3), np.uint8),
            'object_id': None
        })
        
    def get_best(self, camera, label):
        if label in self.camera_data[camera]['best_objects']:
            return self.camera_data[camera]['best_objects'][label]['frame']
        else:
            return None
    
    def get_current_frame(self, camera):
        return self.camera_data[camera]['current_frame']

    def run(self):
        while True:
            camera, frame_time, tracked_objects = self.tracked_objects_queue.get()

            config = self.config[camera]
            best_objects = self.camera_data[camera]['best_objects']
            current_object_status = self.camera_data[camera]['object_status']
            self.camera_data[camera]['tracked_objects'] = tracked_objects

            ###
            # Draw tracked objects on the frame
            ###
            object_id_hash = hashlib.sha1(str.encode(f"{camera}{frame_time}"))
            object_id_bytes = object_id_hash.digest()
            object_id = plasma.ObjectID(object_id_bytes)
            current_frame = self.plasma_client.get(object_id, timeout_ms=0)

            if not current_frame is plasma.ObjectNotAvailable:
                # draw the bounding boxes on the frame
                for obj in tracked_objects.values():
                    thickness = 2
                    color = COLOR_MAP[obj['label']]
                    
                    if obj['frame_time'] != frame_time:
                        thickness = 1
                        color = (255,0,0)

                    # draw the bounding boxes on the frame
                    box = obj['box']
                    draw_box_with_label(current_frame, box[0], box[1], box[2], box[3], obj['label'], f"{int(obj['score']*100)}% {int(obj['area'])}", thickness=thickness, color=color)
                    # draw the regions on the frame
                    region = obj['region']
                    cv2.rectangle(current_frame, (region[0], region[1]), (region[2], region[3]), (0,255,0), 1)
                
                if config['snapshots']['show_timestamp']:
                    time_to_show = datetime.datetime.fromtimestamp(frame_time).strftime("%m/%d/%Y %H:%M:%S")
                    cv2.putText(current_frame, time_to_show, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=.8, color=(255, 255, 255), thickness=2)

                ###
                # Set the current frame as ready
                ###
                self.camera_data[camera]['current_frame'] = current_frame

                # store the object id, so you can delete it at the next loop
                previous_object_id = self.camera_data[camera]['object_id']
                if not previous_object_id is None:
                    self.plasma_client.delete([previous_object_id])
                self.camera_data[camera]['object_id'] = object_id
            
            ###
            # Maintain the highest scoring recent object and frame for each label
            ###
            for obj in tracked_objects.values():
                # if the object wasn't seen on the current frame, skip it
                if obj['frame_time'] != frame_time:
                    continue
                if obj['label'] in best_objects:
                    now = datetime.datetime.now().timestamp()
                    # if the object is a higher score than the current best score 
                    # or the current object is more than 1 minute old, use the new object
                    if obj['score'] > best_objects[obj['label']]['score'] or (now - best_objects[obj['label']]['frame_time']) > 60:
                        obj['frame'] = np.copy(self.camera_data[camera]['current_frame'])
                        best_objects[obj['label']] = obj
                else:
                    obj['frame'] = np.copy(self.camera_data[camera]['current_frame'])
                    best_objects[obj['label']] = obj

            ###
            # Report over MQTT
            ###
            # count objects with more than 2 entries in history by type
            obj_counter = Counter()
            for obj in tracked_objects.values():
                if len(obj['history']) > 1:
                    obj_counter[obj['label']] += 1
                    
            # report on detected objects
            for obj_name, count in obj_counter.items():
                new_status = 'ON' if count > 0 else 'OFF'
                if new_status != current_object_status[obj_name]:
                    current_object_status[obj_name] = new_status
                    self.client.publish(f"{self.topic_prefix}/{camera}/{obj_name}", new_status, retain=False)
                    # send the best snapshot over mqtt
                    best_frame = cv2.cvtColor(best_objects[obj_name]['frame'], cv2.COLOR_RGB2BGR)
                    ret, jpg = cv2.imencode('.jpg', best_frame)
                    if ret:
                        jpg_bytes = jpg.tobytes()
                        self.client.publish(f"{self.topic_prefix}/{camera}/{obj_name}/snapshot", jpg_bytes, retain=True)
                    #Save a copy of the best image to local disk
                    today = datetime.datetime.now()
                    year = today.strftime("%Y")
                    month = today.strftime("%m")
                    day = today.strftime("%d")
                    time = today.strftime("%H%M%S")
                    output = "/storage/"  + year + "/" + month + "/" + day + "/" + obj_name + "/"
                    pathlib.Path(output).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(output + time + ".png", best_frame)
                    cv2.imwrite("/storage/" + obj_name + ".png", best_frame)
                    if self.camera.name == "front":
                        camera_port = 8084
                    elif self.camera.name == "backdeck":
                        camera_port = 8082
                    elif self.camera.name == "backyard":
                        camera_port = 8081
                    elif self.camera.name == "frontdoor":
                        camera_port = 8089
                    elif self.camera.name == "laundry":
                        camera_port = 8087
                    elif self.camera.name == "laundryback":
                        camera_port = 8083
                    elif self.camera.name == "underhouse":
                        camera_port = 8085
                    else:
                        print("Unable to convert camera name to port for:" + self.camera.name)
                    print("Object Detected - Type:{} with Confidence:{}% on Camera:{}".format(obj_name,int(obj['score']*100),self.camera.name))
                    # Notify Motion that an event has started
                    req_url = "http://192.168.11.144:7999/" + str(camera_port) + "/action/eventstart"
                    print("DEBUG: Parsed Motion API Url is: " + str(req_url) )
                    response = requests.get(req_url)
                    print("Start Recording Request sent to: " + self.camera.name)
                    dis_url = "http://192.168.11.144:7999/" + str(camera_port) + "/detection/pause"
                    response = requests.get(dis_url)
            # expire any objects that are ON and no longer detected
            expired_objects = [obj_name for obj_name, status in current_object_status.items() if status == 'ON' and not obj_name in obj_counter]
            for obj_name in expired_objects:
                current_object_status[obj_name] = 'OFF'
                self.client.publish(f"{self.topic_prefix}/{camera}/{obj_name}", 'OFF', retain=False)
                # send updated snapshot over mqtt
                best_frame = cv2.cvtColor(best_objects[obj_name]['frame'], cv2.COLOR_RGB2BGR)
                ret, jpg = cv2.imencode('.jpg', best_frame)
                if ret:
                    jpg_bytes = jpg.tobytes()
                    self.client.publish(f"{self.topic_prefix}/{camera}/{obj_name}/snapshot", jpg_bytes, retain=True)
                    if self.camera.name == "front":
                        camera_port = 8084
                    elif self.camera.name == "backdeck":
                        camera_port = 8082
                    elif self.camera.name == "backyard":
                        camera_port = 8081
                    elif self.camera.name == "frontdoor":
                        camera_port = 8089
                    elif self.camera.name == "laundry":
                        camera_port = 8087
                    elif self.camera.name == "laundryback":
                        camera_port = 8083
                    elif self.camera.name == "underhouse":
                        camera_port = 8085
                    else:
                        print("Unable to convert camera name to port for:" + self.camera.name)
                    # Notify Motion that an even has ended
                    req_url = "http://192.168.11.144:7999/" + str(camera_port) + "/action/eventend"
                    print("DEBUG: Parsed Motion API Url is: " + str(req_url) )
                    response = requests.get(req_url)
                    print("End Recording Request successfully sent to: " + self.camera.name)                        
