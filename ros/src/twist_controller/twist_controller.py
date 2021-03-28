import rospy
import numpy as np
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

class Controller(object):
    def __init__(self, vehicle_mass, wheel_radius, wheel_base, steer_ratio, min_speed,
                         max_lat_accel, max_steer_angle, accel_limit, decel_limit):
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        self.max_brake_torque = vehicle_mass * decel_limit * wheel_radius
        self.decel_limit = decel_limit
        # initialize the controllers
        self.speed_controller = PID(0.3, 0.1, 0, 0, 0.2)
        self.steering_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        # low pass filter for speeed
        tau = 0.5 # cut-off frequency
        ts = 0.02 # 1/sample frequency 50Hz
        self.low_pass_filter = LowPassFilter(tau , ts)        
        self.last_time = None
        

    def control(self, target_angular_velocity, target_linear_velocity, 
                current_angular_velocity, current_linear_velocity, dbw_enabled):
        steering = 0.0
        throttle = 0.0
        brake = 0.0
        # return 0 if dbw is not enabled. Safety driver takes control
        if not dbw_enabled:
            self.speed_controller.reset()
            return throttle, brake, steering
        
        # calculate steering values
        current_linear_velocity = self.low_pass_filter.filt(current_linear_velocity) 
        steering = self.steering_controller.get_steering(target_linear_velocity, target_angular_velocity, current_linear_velocity)
            
        # calculate delta t
        current_time = rospy.get_time()
        dt =  (current_time - self.last_time) if self.last_time else 0.02
        self.last_time = current_time
           
        # calculate throttle/brake 
        error = target_linear_velocity - current_linear_velocity
        throttle = self.speed_controller.step(error, dt)
        brake = 0

        if target_linear_velocity == 0 and current_linear_velocity < 0.1:
            # full brake to stop the vehicle
            throttle = 0
            brake = 400 # hint from the project notes            
        elif throttle < 0.1 and error < 0:
            # decelerate inline with the PID error or the vehicle decl. limit whichever is maximum
            throttle = 0
            deceleration = max (error, self.decel_limit)
            brake = self.vehicle_mass * abs(deceleration) * self.wheel_radius
            
        return throttle, brake, steering
