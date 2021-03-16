#!/usr/bin/env python

import rospy
import math
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
Based on the code walkthrough lesson by Stephen and Aaron
'''

LOOKAHEAD_WPS = 200 # Number of waypoints after the current position which will be published

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        
        # Subscribing to vehicles current position
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        
        # Subscribing to a list of all waypoints for the track
        # This list includes waypoints both before and after the vehicle. 
        # The publisher send this information only once.
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # member variables for processing sub, pub information
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.final_waypoints = None
        self.traffic_waypoint_index = -1
        
        # Instead of rospy.spin() , sleep() is called to publish final waypoints  
        # at regular intervals
        self.loop()
    
    def loop(self):
        """
        Main control loop. the published waypoints will be consumed by Autoware (Waypoint follower) which 
        runs at 30 Hz.
        """
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoints_tree:
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()
    
    def get_closest_waypoint_idx(self):
        '''
        Returns the closest waypoint w.r.t to the current vehicle position  
        '''
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoints_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx
     
    def publish_waypoints(self, closest_idx):
        '''
        Slices a set of waypoints from the base_waypoints array. 
        Starting point is the closest position of the vehicle to the
        Ending point is 'LOOKAHEAD_WPS' number away from the closest point  
        '''
        lane = Lane()
        lane.waypoints = self.base_waypoints.waypoints[closest_idx: closest_idx+LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish(lane)
        
    def pose_cb(self, msg):
        '''
        callback handler for the ROS topic: /current_pose 
        msg: PoseStamped - indicates the current position of the vehicle
        msg.header - timestamped data, uniquely identified data with a frame-id
        msg.pose - representation of pose in free space, composed of position and orientation in free space in quaternion form 
        '''
        self.pose = msg
    
    def waypoints_cb(self, waypoints):
        '''
        callback handler for the ROS topic: /base_waypoints 
        msg: Lane - indicates the positions of all waypoints in the current driving environment
        msg.header - timestamped data, uniquely identified data with a frame-id
        msg.waypoints - array of waypoint, each representing :
        msg.waypoints[i].pose - time-stamped position orientation in free space in quaternion form 
        msg.waypoints[i].twist - time-stamped velocity in free space broken into its linear and angular parts.
        '''
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            rospy.loginfo("Waypoint Tree Initialited. Total count of waypoints : {}".format(len(self.waypoints_2d)))
            self.waypoints_tree = KDTree(self.waypoints_2d)
            
    def traffic_cb(self, msg):
        '''
        callback handler for the ROS topic: /traffic_waypoint 
        msg: Int32 - indicates the index of the next traffic waypoint
        '''
        self.traffic_waypoint_index = msg.data;
        
    
    def obstacle_cb(self, msg):
        '''
        callback handler for the ROS topic: /obstacle_waypoints 
        msg: Not implemented
        '''
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
