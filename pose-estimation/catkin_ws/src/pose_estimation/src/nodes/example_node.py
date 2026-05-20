#!/usr/bin/env python3

import rospy
from std_msgs.msg import String


class ExampleNode:
    def __init__(self):
        rospy.init_node("example_node")

        self.publish_topic = "/example_out"
        self.subscribe_topic = "/example_in"
        self.timer_period = 1.0

        # Publisher
        self.pub = rospy.Publisher(self.publish_topic, String, queue_size=10)

        # Subscriber
        self.sub = rospy.Subscriber(self.subscribe_topic, String, self.callback)

        # Timer
        self.timer = rospy.Timer(rospy.Duration(self.timer_period), self.timer_callback)

        self.last_msg = None

        rospy.loginfo("ExampleNode initialized")
        rospy.loginfo(f"Publishing to: {self.publish_topic}")
        rospy.loginfo(f"Subscribing to: {self.subscribe_topic}")

    def callback(self, msg: String):
        """Callback dla subskrybera."""
        rospy.loginfo(f"Received: {msg.data}")
        self.last_msg = msg.data

    def timer_callback(self, event):
        """Wywoływane cyklicznie przez timer."""
        if self.last_msg is None:
            output = "No message received yet"
        else:
            output = f"Last received: {self.last_msg}"

        msg = String()
        msg.data = output

        self.pub.publish(msg)
        rospy.loginfo(f"Published: {msg.data}")


def main():
    try:
        node = ExampleNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()