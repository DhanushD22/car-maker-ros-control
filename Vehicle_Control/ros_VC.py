#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.time import Time

from pycarmaker import CarMaker, Quantity


class VehicleController(Node):
    def __init__(self):
        super().__init__('carmaker_ros_control')
        self.get_logger().info("CarMaker ROS Speed Controller Starting...")

        # -------------------------------------------------
        # Connect to CarMaker
        # -------------------------------------------------
        self.cm = CarMaker("192.168.0.162", 16660)
        self.cm.connect()
        self.get_logger().info("Connected to IPG CarMaker")

        # -------------------------------------------------
        # Quantities
        # -------------------------------------------------
        self.q_lat_passive  = Quantity("Driver.Lat.passive",  Quantity.INT)
        self.q_long_passive = Quantity("Driver.Long.passive", Quantity.INT)

        self.q_gas       = Quantity("VC.Gas",   Quantity.FLOAT)
        self.q_brake     = Quantity("VC.Brake", Quantity.FLOAT)
        self.q_gear      = Quantity("VC.GearNo", Quantity.INT)
        self.q_velocity  = Quantity("Car.v", Quantity.FLOAT)

        for q in (
            self.q_lat_passive,
            self.q_long_passive,
            self.q_gas,
            self.q_brake,
            self.q_gear,
            self.q_velocity
        ):
            self.cm.subscribe(q)

        # -------------------------------------------------
        # Control parameters (TUNE THESE)
        # -------------------------------------------------
        self.KF_GAS   = 0.03
        self.KP_GAS   = 0.08
        self.KP_BRAKE = 0.10
        self.MAX_GAS  = 1.0
        self.MAX_BRAKE = 1.0

        # Deadband and hold parameters
        self.ERROR_DEADBAND = 0.10   # ±0.1 m/s tolerance
        self.STEADY_GAS = 0.08       # Constant throttle to maintain speed
        
        # Brake hold parameters
        self.BRAKE_HOLD_VALUE = 1.0      # Full brake when stopped
        self.STOPPED_THRESHOLD = 0.05    # Vehicle considered stopped below this velocity (m/s)

        self.desired_velocity = 0.0
        self.last_cmd_time = None
        self.CONTROL_TIMEOUT = 1.0   # seconds – after this with no cmd_vel → IPG takes over

        # -------------------------------------------------
        # ROS - PUBLISHER for toy car
        # -------------------------------------------------
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscription for commands from traffic light detection
        self.create_subscription(Twist, '/cmd_vel_command', self.cmdvel_callback, 10)

        # DEFAULT: Full IPGDriver control (both lateral + longitudinal)
        self.cm.DVA_write(self.q_lat_passive, 0)   # IPG always steers (follows maneuver/path)
        self.cm.DVA_write(self.q_long_passive, 0)  # IPG controls gas/brake/gear by default

        self.create_timer(0.02, self.control_loop)  # 50 Hz

    # -------------------------------------------------
    def cmdvel_callback(self, msg: Twist):
        """Receives command from traffic light detection script"""
        self.desired_velocity = msg.linear.x
        self.last_cmd_time = self.get_clock().now()
        self.get_logger().info(f"Received command: {self.desired_velocity:.2f} m/s")

    # -------------------------------------------------
    def publish_velocity_to_toy_car(self, velocity):
        """Always publish velocity to toy car - either ROS command or CarMaker actual velocity"""
        msg = Twist()
        msg.linear.x = velocity
        self.cmd_vel_publisher.publish(msg)

    # -------------------------------------------------
    def control_loop(self):
        self.cm.read()
        actual_velocity = self.q_velocity.data or 0.0

        now = self.get_clock().now()
        
        # Determine if we have a recent command
        in_ros_mode = False
        if self.last_cmd_time is not None:
            elapsed = (now - self.last_cmd_time).nanoseconds / 1e9
            if elapsed < self.CONTROL_TIMEOUT:
                in_ros_mode = True

        if in_ros_mode:
            # ROS override active - control simulation AND publish to toy car
            self.cm.DVA_write(self.q_long_passive, 1)
            self.cm.DVA_write(self.q_gear, 1)

            # Check if desired velocity is ZERO (stop command)
            if abs(self.desired_velocity) < 0.01:
                # Apply full brake to stop and hold
                self.cm.DVA_write(self.q_gas, 0.0)
                self.cm.DVA_write(self.q_brake, self.BRAKE_HOLD_VALUE)
                
                # Publish ZERO to toy car
                self.publish_velocity_to_toy_car(0.0)
                
                if abs(actual_velocity) < self.STOPPED_THRESHOLD:
                    self.get_logger().info(
                        f"🔴 ROS MODE - STOPPED & HOLDING: vel={actual_velocity:.3f} m/s, "
                        f"brake={self.BRAKE_HOLD_VALUE}, toy_car_cmd=0.0"
                    )
                else:
                    self.get_logger().info(
                        f"🟡 ROS MODE - BRAKING TO STOP: vel={actual_velocity:.3f} m/s, "
                        f"brake={self.BRAKE_HOLD_VALUE}, toy_car_cmd=0.0"
                    )
            
            else:
                # Normal velocity control (positive desired velocity)
                error = self.desired_velocity - actual_velocity
                gas = 0.0
                brake = 0.0

                if abs(error) < self.ERROR_DEADBAND:
                    if self.desired_velocity > 0.3:
                        gas = self.STEADY_GAS
                else:
                    if error > 0:
                        gas = self.KF_GAS * self.desired_velocity + self.KP_GAS * error
                        gas = min(max(gas, 0.0), self.MAX_GAS)
                    else:
                        brake = min(self.KP_BRAKE * (-error), self.MAX_BRAKE)

                self.cm.DVA_write(self.q_gas, gas)
                self.cm.DVA_write(self.q_brake, brake)
                
                # Publish desired velocity to toy car
                self.publish_velocity_to_toy_car(self.desired_velocity)
                
                self.get_logger().info(
                    f"🟢 ROS MODE - NORMAL CONTROL: target={self.desired_velocity:.2f}, "
                    f"actual={actual_velocity:.2f}, gas={gas:.3f}, brake={brake:.3f}, "
                    f"toy_car_cmd={self.desired_velocity:.2f}"
                )

        else:
            # CarMaker has control - publish CarMaker's actual velocity to toy car
            self.cm.DVA_write(self.q_long_passive, 0)

            # Set neutral values
            self.cm.DVA_write(self.q_gas,   0.0)
            self.cm.DVA_write(self.q_brake, 0.0)

            # Release ALL active DVA overrides
            self.cm.DVA_release()

            # CRITICAL: Publish CarMaker's actual velocity to toy car
            self.publish_velocity_to_toy_car(actual_velocity)

            # Logging on transition (prevents spamming)
            if hasattr(self, '_was_in_ros_mode') and self._was_in_ros_mode:
                self.get_logger().info(
                    f"⚪ CARMAKER MODE: IPGDriver in control. "
                    f"Publishing Car.v={actual_velocity:.2f} m/s to toy car"
                )
                self._was_in_ros_mode = False
            elif not hasattr(self, '_was_in_ros_mode'):
                # First time in CarMaker mode
                self.get_logger().info(
                    f"⚪ CARMAKER MODE: Publishing Car.v={actual_velocity:.2f} m/s to toy car"
                )

        self._was_in_ros_mode = in_ros_mode

def main(args=None):
    rclpy.init(args=args)
    node = VehicleController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Gentle shutdown – give control back to IPG and stop toy car
        node.cm.DVA_write(node.q_long_passive, 0)
        node.cm.DVA_write(node.q_gas, 0.0)
        node.cm.DVA_write(node.q_brake, 0.0)
        node.cm.DVA_release()
        node.cm.read()
        
        # Stop toy car
        node.publish_velocity_to_toy_car(0.0)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



# code changed to subscribe to /cmd_vel_command instead of /cmd_vel to avoid conflict with toy car publisher and 
# added publishing actual velocity to toy car when IPG has control.