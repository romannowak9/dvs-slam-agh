#!/usr/bin/env python3

import rospy
import numpy as np
from dvs_msgs.msg import EventArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class EventAggregatorNode:
    def __init__(self):
        rospy.init_node('event_aggregator_node')

        # Pobieranie parametrów
        self.frame_type = rospy.get_param('~frame_type', 'standard')
        self.do_filter = rospy.get_param('~do_filter', False)
        self.filter_t = rospy.get_param('~filter_t', 0.05)
        self.width = rospy.get_param('~width', 640)
        self.height = rospy.get_param('~height', 480)
        
        self.img_shape = (self.height, self.width)
        self.bridge = CvBridge()
        
        # SAE (Surface of Active Events) for filtration
        self.sae = np.zeros(self.img_shape, dtype=np.float64)
        
        self.event_sub = rospy.Subscriber('/dvxplorer_left/events', EventArray, self.callback)
        self.image_pub = rospy.Publisher('/event_frames', Image, queue_size=1)

        rospy.loginfo(f"Evnets Aggregator Node Started. Mode: {self.frame_type}")

    def callback(self, msg):
        if not msg.events:
            return

        xs = np.array([e.x for e in msg.events], dtype=np.uint16)
        ys = np.array([e.y for e in msg.events], dtype=np.uint16)
        ps = np.array([e.polarity for e in msg.events], dtype=bool)
        ts = np.array([e.ts.to_sec() for e in msg.events], dtype=np.float64)

        # 1. BAF Filrtration
        if self.do_filter:
            # Time from last event in this pixel
            mask = (ts - self.sae[ys, xs]) < self.filter_t
            self.sae[ys, xs] = ts
            xs, ys, ps, ts = xs[mask], ys[mask], ps[mask], ts[mask]
            if len(xs) == 0: return

        # 2. Generowanie ramki
        if self.frame_type == "exponential":
            frame = self.generate_exponential_frame(xs, ys, ps, ts)
        else:
            frame = self.generate_standard_frame(xs, ys, ps)

        # 3. Publikacja
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="mono8")
        img_msg.header = msg.header
        self.image_pub.publish(img_msg)

    def generate_standard_frame(self, xs, ys, ps):
        # Gray background
        frame = np.full(self.img_shape, 127, dtype=np.uint8)
        
        frame[ys[ps == False], xs[ps == False]] = 0
        frame[ys[ps == True], xs[ps == True]] = 255
        return frame

    def generate_exponential_frame(self, xs, ys, ps, ts):
        t_end = ts[-1]
        dt = t_end - ts[0]
        if dt <= 0: dt = 1e-6
        
        decay = np.exp(-(t_end - ts) / dt)
        
        # Polarization values: 1; -1
        p_vals = np.where(ps, 1.0, -1.0)
        
        intensities = (p_vals * decay + 1.0) * 127.5
        intensities = np.clip(intensities, 0, 255).astype(np.uint8)
        
        frame = np.full(self.img_shape, 127, dtype=np.uint8)
        frame[ys, xs] = intensities
        return frame

if __name__ == '__main__':
    try:
        node = EventAggregatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass