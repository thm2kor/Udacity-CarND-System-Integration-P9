#!/usr/bin/env python

import rospy
import math
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int32, Float32, Float32MultiArray
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from itertools import cycle, islice

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
Based on the code walkthrough lesson by Stephen Welch and Aaron Brown
'''

LOOKAHEAD_WPS = 100 # Number of waypoints after the current position which will be published
MAX_DECEL = 0.5         # Max decleration during the "DIST_APPLY_BRAKING" phase
DIST_BRAKE_START = 15   # Stopping distance after which braking starts
DIST_HARD_BRAKING = 5   # Stopping distance after which hard braking starte
DIST_APPLY_BRAKING = 10 # Stopping distance after which gradual braking starte
CONST_ACCLERATION = 2   # Acceleration factor when the vehicle is cruising
SPIN_FREQUENCY = 20     # Frequency at which the waypoints will be updated

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')
        
        # Subscribing to vehicles current position
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        # Subscribing to a list of all waypoints for the track
        # This list includes waypoints both before and after the vehicle. 
        # The publisher send this information only once.
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        # Add a subscriber for /traffic_waypoint, current and target velocity 
        rospy.Subscriber('/traffic_waypoint', Float32MultiArray, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/target_velocity', Float32, self.target_velocity_cb)
        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)
        self.closest_waypoint_pub = rospy.Publisher('/closest_waypoint', Int32, queue_size=1)
        
        # member variables for processing sub, pub information
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.final_waypoints = None
        self.traffic_waypoint_index = -1
        self.traffic_waypoint_x = -1
        self.traffic_waypoint_y = -1  
        self.current_velocity = 0
        # Instead of rospy.spin() , sleep() is called to publish final waypoints  
        # at controlled intervals
        self.loop()
    
    def loop(self):
        """
        Main control loop. the published waypoints will be consumed by Autoware (Waypoint follower) which 
        runs at SPIN_FREQUENCY Hz.
        """
        rate = rospy.Rate(SPIN_FREQUENCY)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints and self.waypoints_tree:
                # get the index of the closest waypoint next to the vehicles
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                # publish LOOKAHEAD_WPS number of waypoints
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()
            
    def get_closest_waypoint_idx(self):
        '''
        Returns the closest waypoint w.r.t to the current vehicle position  
        Based on the code walkthrough lesson by Stephen Welch and Aaron Brown
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

        self.closest_waypoint_pub.publish(closest_idx)
        return closest_idx
    
   
    def publish_waypoints(self, closest_idx):
        '''
        Slices a set of waypoints from the base_waypoints array. 
        Starting point is the closest position of the vehicle to the
        Ending point is 'LOOKAHEAD_WPS' number away from the closest point  
        '''
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        
        # keep looping around the track
        if (closest_idx+1) % len(self.waypoints_2d) == 0:
            closest_idx = 0
        # prepare a separate list of waypoints to be published
        sliced_waypoints = list(islice(cycle(self.base_waypoints.waypoints), closest_idx, closest_idx + LOOKAHEAD_WPS))
        stop_line_dist = math.sqrt((self.pose.pose.position.x-self.traffic_waypoint_x)**2 + 
                                   (self.pose.pose.position.y-self.traffic_waypoint_y)**2)
        # Instead of stop_line_dist, an offset of closest_idx could have been used. But i prefered the
        # distance calculation method since it offers more control of the deceleration logic
        
        # Set the velocities of the waypoints depending on the traffic conditions
        if (self.traffic_waypoint_index == -1) or (stop_line_dist >= DIST_BRAKE_START) :
            # set velocities for constant acceleration if there is no RED traffic signal or 
            # if the vehicle is far away from a traffic signal
            lane.waypoints = self.accelerate_waypoints(sliced_waypoints)    
        else:
            # set velocities for decleration
            lane.waypoints = self.decelerate_waypoints(sliced_waypoints, closest_idx)            
        
        self.final_waypoints_pub.publish(lane)
        
    
    def pose_cb(self, msg):
        '''
        callback handler for the ROS topic: /current_pose 
        msg: PoseStamped - indicates the current position of the vehicle
        msg.header - timestamped data, uniquely identified data with a frame-id
        msg.pose - representation of pose in free space, composed of position and orientation in free space in quaternion form 
        '''
        self.pose = msg        
    
    def velocity_cb(self, msg):
        '''
        callback handler for the ROS topic: /current_velocity 
        msg: TwistStamped - indicates the current velocity of the vehicle
        msg.header - timestamped data, uniquely identified data with a frame-id
        msg.twist - representation of velocity in free space broken into its linear and angular parts.
        msg.twist.linear - linear component for the (x,y,z) velocities
        msg.twist.angular -  angular rate about the (x,y,z) axes
        '''
        self.current_velocity = msg.twist.linear.x
    
    def target_velocity_cb(self, msg): 
        '''
        callback handler for the ROS topic: /target_velocity 
        msg: Target velocity (Float32) published from waypoint_loader
        '''
        self.target_velocity = msg.data
        
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
            rospy.loginfo("Waypoint tree initialited. Total count of waypoints : {}".format(len(self.waypoints_2d)))
            self.waypoints_tree = KDTree(self.waypoints_2d)
            
    def traffic_cb(self, msg):
        '''
        callback handler for the ROS topic: /traffic_waypoint 
        msg: Float32MultiArray- Index [0] - indicates the index of the next traffic waypoint
        msg: Float32MultiArray- Index [1] - indicates the X coordinate of the next traffic waypoint
        msg: Float32MultiArray- Index [2] - indicates the y coordinate of the next traffic waypoint
        '''
        self.traffic_waypoint_index = int(msg.data[0])        
        self.traffic_waypoint_x = msg.data[1]
        self.traffic_waypoint_y = msg.data[2]
 
        
    def obstacle_cb(self, msg):
        '''
        callback handler for the ROS topic: /obstacle_waypoints 
        msg: Not implemented
        '''
        pass

    def get_waypoint_velocity(self, waypoint):
        '''
        Returns the given velocity to the <waypoint> index within the <waypoints> list      
        '''
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        '''
        Sets the given velocity to the <waypoint> index within the <waypoints> list      
        '''
        waypoints[waypoint].twist.twist.linear.x = velocity
    
    def accelerate_waypoints(self, waypoints):
        '''
        Sets the velocity of the waypoints so that the vehicle accelerates smoothly within
        the given limits of deceleration and jerk
        Based on the code walkthrough lesson by Stephen Welch and Aaron Brown        
        '''
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

        for i in range(len(waypoints)):
            wp_prev = waypoints[i-1].pose.pose.position
            wp_now = waypoints[i].pose.pose.position
            set_velocity = min(self.target_velocity, 
                               math.sqrt(self.current_velocity*self.current_velocity + 2*CONST_ACCLERATION*dl(wp_prev, wp_now)))
            self.set_waypoint_velocity(waypoints, i, set_velocity)

        return waypoints
    
    def decelerate_waypoints(self, waypoints, closest_idx):
        '''
        Sets the velocity of the waypoints so that the vehicle declerates smoothly within
        the given limits of deceleration and jerk
        Based on the code walkthrough lesson by Stephen Welch and Aaron Brown        
        '''
        result = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            # Distance includes a number of waypoints back so front of car stops at line            
            stop_line_dist = math.sqrt((self.pose.pose.position.x-self.traffic_waypoint_x)**2 + 
                                   (self.pose.pose.position.y-self.traffic_waypoint_y)**2)
            # braking in three stages
            if stop_line_dist <= DIST_HARD_BRAKING:
                velocity = 0
            elif stop_line_dist <= DIST_APPLY_BRAKING: # velocity slopes down like a quadratic equation
                velocity = math.sqrt(2 * MAX_DECEL * stop_line_dist)
            else:
                velocity = wp.twist.twist.linear.x - (wp.twist.twist.linear.x/stop_line_dist)
            
            if velocity < 1.0:
                velocity = 0.0 
            p.twist.twist.linear.x = min(velocity, wp.twist.twist.linear.x)
            result.append(p)          
        
        return result
    

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
