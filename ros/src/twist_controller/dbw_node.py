#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd
from geometry_msgs.msg import TwistStamped
from twist_controller import Controller

''' 
Once messages are being published to /final_waypoints, the vehicle's waypoint follower will publish twist commands
to the /twist_cmd topic.This node will subscribe to /twist_cmd and use various controllers to provide appropriate 
throttle, brake, and steering commands. These commands can then be published to the following topics:

    /vehicle/throttle_cmd
    /vehicle/brake_cmd
    /vehicle/steering_cmd
    
It is possible that a safety driver may take control of the car during testing in a realcar. If this happens,
the PID controllers will start to accumulate error. To avoid this scenario, the status of DBW (drive-by-wire) is always 
checked before publishing the actuator commands. The DBW status is determined by subscribing to /vehicle/dbw_enabled. 

This node is currently set up to publish steering, throttle, and brake commands at 50hz. 
The DBW system on Carla expects messages at this frequency, and will disengage (reverting control back to the driver) 
if control messages are published at less than 10hz.

The code is based on the code walk through by Aaron and Stephen.
'''

class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        #fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        #brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        min_speed = 0. # rospy.get_param('~min_speed', 0.1)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
        
        # Initializing the publishers
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        # controller which calculates the required throttle, brake and steering
        self.controller = Controller(vehicle_mass, wheel_radius, wheel_base, steer_ratio, min_speed,
                                     max_lat_accel, max_steer_angle, accel_limit, decel_limit)

        # Subscription to the relevant ROS Nodes
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb)
        
        # Members will be set in the callback functions
        self.current_linear_velocity = None
        self.current_angular_velocity = None
        self.target_linear_velocity = None
        self.target_angular_velocity = None
        self.dbw_enabled = False
        self.throttle = 0.
        self.steering = 0.
        self.brake = 0.
        
        # control the publishing frequency @ 50Hz
        self.loop()
        
    def loop(self):
        rate = rospy.Rate(50) # spin at a frequency of 50 hz
        while not rospy.is_shutdown():
            # calculate the actuator commands only when the member variables are set by the callback funcitons
            if not None in (self.target_angular_velocity, self.target_linear_velocity, 
                            self.current_angular_velocity, self.current_linear_velocity):
                throttle, brake, steering = self.controller.control(self.target_angular_velocity,
                                    self.target_linear_velocity, self.current_angular_velocity,
                                    self.current_linear_velocity, self.dbw_enabled)
                # publish only when the drive by wire functionality is enabled
                if self.dbw_enabled:                
                    self.publish(throttle, brake, steering)
            rate.sleep()
    
    def velocity_cb(self, msg):
        '''
        callback handler for the ROS topic:/current_velocity 
        msg : geometry_msgs.msg.TwistStamped
        '''
        self.current_linear_velocity = msg.twist.linear.x
        self.current_angular_velocity = msg.twist.angular.z

    def dbw_enabled_cb(self, msg):
        '''
        callback handler for the ROS topic: /vehicle/dbw_enabled
        msg: Bool- True if the the car is under DBW control. False if a safety driver is controlling the car
        '''
        if self.dbw_enabled != msg.data:
            rospy.loginfo('DBW Status changed : old [{}] ==> new [{}]'.format(self.dbw_enabled, msg.data))
        self.dbw_enabled = msg.data
        
    def twist_cb(self, msg):
        '''
        callback handler for the ROS topic: /twist_cmd 
        msg : geometry_msgs.msg.TwistStamped
        '''
        self.target_linear_velocity = msg.twist.linear.x
        self.target_angular_velocity = msg.twist.angular.z
        
    def publish(self, throttle, brake, steer):
        rospy.loginfo('Publishing ... [throttle ==> {:5.4f}, brake ==> {:5.4f}, steer ==> {:5.4f}]'.format(throttle, brake, steer))
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)
        
        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)

if __name__ == '__main__':
    DBWNode()
