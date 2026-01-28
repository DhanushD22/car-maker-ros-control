#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from pycarmaker import CarMaker, Quantity


class VehicleController(Node):
    def __init__(self):
        super().__init__('carmaker_ros_control')
        self.get_logger().info("CarMaker ROS Integrated Controller Starting...")

        # Connect to CarMaker
        self.cm = CarMaker("192.168.0.162", 16660)
        self.cm.connect()
        self.get_logger().info("Connected to IPG CarMaker")

        # Quantities
        self.q_lat_passive  = Quantity("Driver.Lat.passive",  Quantity.INT)
        self.q_long_passive = Quantity("Driver.Long.passive", Quantity.INT)
        self.q_gas       = Quantity("VC.Gas",   Quantity.FLOAT)
        self.q_brake     = Quantity("VC.Brake", Quantity.FLOAT)
        self.q_gear      = Quantity("VC.GearNo", Quantity.INT)
        self.q_velocity  = Quantity("Car.v", Quantity.FLOAT)

        for q in (self.q_lat_passive, self.q_long_passive, self.q_gas, 
                  self.q_brake, self.q_gear, self.q_velocity):
            self.cm.subscribe(q)

        # Control parameters
        self.KF_GAS   = 0.03
        self.KP_GAS   = 0.08
        self.KP_BRAKE = 0.10
        self.MAX_GAS  = 1.0
        self.MAX_BRAKE = 1.0
        self.ERROR_DEADBAND = 0.10
        self.STEADY_GAS = 0.08
        self.BRAKE_HOLD_VALUE = 1.0
        self.STOPPED_THRESHOLD = 0.05

        # Toy car velocity limits
        self.MAX_TOY_CAR_VELOCITY = 2.0
        self.MIN_TOY_CAR_VELOCITY = 0.7
        self.CAR_MOVING_THRESHOLD = 0.1  # Car.v must be > 0.1 to apply min velocity

        # Control inputs (unified object avoidance)
        self.traffic_light_cmd = None
        self.object_avoid_cmd = None
        self.object_avoid_brake = None
        
        # Last command times
        self.last_traffic_time = None
        self.last_object_time = None
        
        self.CONTROL_TIMEOUT = 0.5  # 500ms

        # ROS Subscriptions and Publishers
        self.create_subscription(Twist, '/traffic_light_cmd', self.traffic_light_callback, 10)
        self.create_subscription(Twist, '/object_avoidance_cmd', self.object_avoidance_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # DEFAULT: IPGDriver control
        self.cm.DVA_write(self.q_lat_passive, 0)
        self.cm.DVA_write(self.q_long_passive, 0)

        self._was_in_ros_mode = False
        self.create_timer(0.02, self.control_loop)  # 50 Hz

    def traffic_light_callback(self, msg: Twist):
        self.traffic_light_cmd = msg.linear.x
        self.last_traffic_time = self.get_clock().now()
        self.get_logger().info(f"[TL] Received: {self.traffic_light_cmd:.2f} m/s")

    def object_avoidance_callback(self, msg: Twist):
        self.object_avoid_cmd = msg.linear.x
        self.object_avoid_brake = msg.angular.z
        self.last_object_time = self.get_clock().now()
        self.get_logger().info(f"[OBJ] Received: vel={self.object_avoid_cmd:.2f}, brake={self.object_avoid_brake:.2f}")

    def publish_to_toy_car(self, velocity, control_source="UNKNOWN"):
        """
        Publish velocity to toy car with proper limits:
        - CarMaker mode: Apply 0.7 m/s minimum ONLY when Car.v > 0.1 m/s (actually moving)
        - ROS mode: Use calculated velocity directly
        """
        if control_source == "CARMAKER":
            # CRITICAL FIX: Only apply minimum when car is actually moving
            if velocity > self.CAR_MOVING_THRESHOLD:
                # Car is moving - apply minimum velocity
                if velocity < self.MIN_TOY_CAR_VELOCITY:
                    limited_velocity = self.MIN_TOY_CAR_VELOCITY
                elif velocity > self.MAX_TOY_CAR_VELOCITY:
                    limited_velocity = self.MAX_TOY_CAR_VELOCITY
                else:
                    limited_velocity = velocity
            else:
                # Car is stopped or nearly stopped - publish actual velocity (including 0)
                limited_velocity = min(velocity, self.MAX_TOY_CAR_VELOCITY)
        else:
            # ROS control - use calculated velocity with max limit
            limited_velocity = min(velocity, self.MAX_TOY_CAR_VELOCITY)
        
        msg = Twist()
        msg.linear.x = limited_velocity
        self.cmd_vel_publisher.publish(msg)
        
        return limited_velocity

    def is_command_active(self, last_time):
        if last_time is None:
            return False
        elapsed = (self.get_clock().now() - last_time).nanoseconds / 1e9
        return elapsed < self.CONTROL_TIMEOUT

    def control_loop(self):
        self.cm.read()
        actual_velocity = self.q_velocity.data or 0.0

        # Check active systems
        traffic_active = self.is_command_active(self.last_traffic_time)
        object_active = self.is_command_active(self.last_object_time)
        
        # Priority: Traffic > Object Avoidance > CarMaker
        in_ros_mode = False
        desired_velocity = 0.0
        control_source = "CARMAKER"
        
        if traffic_active:
            # Highest priority: Traffic light
            in_ros_mode = True
            desired_velocity = self.traffic_light_cmd
            control_source = "TRAFFIC"
        elif object_active:
            # Second priority: Object avoidance (unified AEB + ACC)
            in_ros_mode = True
            desired_velocity = self.object_avoid_cmd
            control_source = "OBJECT"
        else:
            # Default: CarMaker
            in_ros_mode = False
            control_source = "CARMAKER"

        if in_ros_mode:
            # ROS override active - stay in control until clear
            self.cm.DVA_write(self.q_long_passive, 1)
            self.cm.DVA_write(self.q_gear, 1)

            # Handle object avoidance with proportional braking
            if control_source == "OBJECT" and self.object_avoid_brake is not None:
                brake_force = self.object_avoid_brake
                
                # Full emergency stop
                if brake_force >= 0.99 or abs(desired_velocity) < 0.01:
                    self.cm.DVA_write(self.q_gas, 0.0)
                    self.cm.DVA_write(self.q_brake, 1.0)
                    toy_vel = self.publish_to_toy_car(0.0, control_source)
                    
                    if abs(actual_velocity) < self.STOPPED_THRESHOLD:
                        self.get_logger().error(f"🔴 [OBJ] EMERGENCY STOP! vel={actual_velocity:.3f}")
                    else:
                        self.get_logger().warning(f"🟠 [OBJ] FULL BRAKE: vel={actual_velocity:.3f} → 0.0")
                
                # Proportional control
                else:
                    error = desired_velocity - actual_velocity
                    gas = 0.0
                    
                    if error > 0 and brake_force < 0.3:
                        # Accelerate gently
                        gas = self.KF_GAS * desired_velocity + self.KP_GAS * error
                        gas = min(max(gas, 0.0), self.MAX_GAS * (1.0 - brake_force))
                        brake = brake_force * 0.5
                    else:
                        # Decelerate
                        gas = 0.0
                        brake = brake_force
                    
                    self.cm.DVA_write(self.q_gas, gas)
                    self.cm.DVA_write(self.q_brake, brake)
                    toy_vel = self.publish_to_toy_car(desired_velocity, control_source)
                    
                    self.get_logger().info(
                        f"🟡 [OBJ] target={desired_velocity:.2f}, actual={actual_velocity:.2f}, "
                        f"brake={brake:.2f}, toy={toy_vel:.2f}"
                    )
            
            # Traffic light stop
            elif abs(desired_velocity) < 0.01:
                self.cm.DVA_write(self.q_gas, 0.0)
                self.cm.DVA_write(self.q_brake, self.BRAKE_HOLD_VALUE)
                toy_vel = self.publish_to_toy_car(0.0, control_source)
                
                if abs(actual_velocity) < self.STOPPED_THRESHOLD:
                    self.get_logger().info(f"🔴 [{control_source}] STOPPED")
                else:
                    self.get_logger().info(f"🟡 [{control_source}] BRAKING → 0.0")
            
            # Normal velocity control
            else:
                error = desired_velocity - actual_velocity
                gas = 0.0
                brake = 0.0

                if abs(error) < self.ERROR_DEADBAND:
                    if desired_velocity > 0.3:
                        gas = self.STEADY_GAS
                else:
                    if error > 0:
                        gas = self.KF_GAS * desired_velocity + self.KP_GAS * error
                        gas = min(max(gas, 0.0), self.MAX_GAS)
                    else:
                        brake = min(self.KP_BRAKE * (-error), self.MAX_BRAKE)

                self.cm.DVA_write(self.q_gas, gas)
                self.cm.DVA_write(self.q_brake, brake)
                toy_vel = self.publish_to_toy_car(desired_velocity, control_source)
                
                self.get_logger().info(
                    f"🟢 [{control_source}] target={desired_velocity:.2f}, "
                    f"actual={actual_velocity:.2f}, toy={toy_vel:.2f}"
                )

        else:
            # CarMaker control
            self.cm.DVA_write(self.q_long_passive, 0)
            self.cm.DVA_write(self.q_gas, 0.0)
            self.cm.DVA_write(self.q_brake, 0.0)
            self.cm.DVA_release()

            # FIXED: Proper velocity publishing for toy car
            toy_vel = self.publish_to_toy_car(actual_velocity, control_source)

            if self._was_in_ros_mode:
                self.get_logger().info(
                    f"⚪ [CARMAKER] Restored. Car.v={actual_velocity:.2f} → toy={toy_vel:.2f}"
                )
                self._was_in_ros_mode = False

        self._was_in_ros_mode = in_ros_mode

def main(args=None):
    rclpy.init(args=args)
    node = VehicleController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cm.DVA_write(node.q_long_passive, 0)
        node.cm.DVA_write(node.q_gas, 0.0)
        node.cm.DVA_write(node.q_brake, 0.0)
        node.cm.DVA_release()
        node.cm.read()
        node.publish_to_toy_car(0.0, "SHUTDOWN")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



# fixed the min velocity to only apply when car is actually moving