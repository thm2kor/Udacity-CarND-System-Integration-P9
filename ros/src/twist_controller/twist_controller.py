import rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter

# GAS_DENSITY = 2.858
# ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, wheel_radius, wheel_base, steer_ratio, min_speed,
                         max_lat_accel, max_steer_angle, accel_limit, decel_limit):
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        self.max_brake_torque = vehicle_mass * decel_limit * wheel_radius
        
        # initialize the controllers
        self.speed_controller = PID(0.2, 0.0001, 0.1, decel_limit, accel_limit)
        self.steering_controller = YawController(wheel_base, steer_ratio, min_speed, 
                                                 max_lat_accel, max_steer_angle)
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
        
        if not dbw_enabled:
            self.speed_controller.reset()
            return steering, throttle, brake
        
        # calculate throttle/brake
        current_time = rospy.get_time()
        if self.last_time is not None:
            dt =  (current_time - self.last_time)
            # rospy.loginfo('Controller delta time - [{:.4f}]'.format(dt))
            error = target_linear_velocity - current_linear_velocity
            acceleration = self.speed_controller.step(error, dt)
            # rospy.loginfo('Unfiltered acceleration  ==> {:.4f}'.format(acceleration))
            
            self.low_pass_filter.filt(acceleration)
            if self.low_pass_filter.ready:
                acceleration = self.low_pass_filter.get() 
                # rospy.loginfo('Filtered acceleration  ==> {:.4f}'.format(acceleration))
                
            if acceleration > 0:
                throttle = acceleration
            else:
                brake = self.vehicle_mass * abs(acceleration) * self.wheel_radius    
                # rospy.loginfo('Brake  ==> {:.4f}'.format(brake))
                
        self.last_time = current_time
        # calculate steering
        steering = self.steering_controller.get_steering(target_linear_velocity, target_angular_velocity, current_linear_velocity)
        
        return throttle, brake, steering
