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
        # ROS
        # -------------------------------------------------
        self.create_subscription(Twist, '/cmd_vel', self.cmdvel_callback, 10)

        # DEFAULT: Full IPGDriver control (both lateral + longitudinal)
        self.cm.DVA_write(self.q_lat_passive, 0)   # IPG always steers (follows maneuver/path)
        self.cm.DVA_write(self.q_long_passive, 0)  # IPG controls gas/brake/gear by default

        self.create_timer(0.02, self.control_loop)  # 50 Hz

    # -------------------------------------------------
    def cmdvel_callback(self, msg: Twist):
        self.desired_velocity = msg.linear.x
        self.last_cmd_time = self.get_clock().now()
        self.get_logger().info(f"Received /cmd_vel: {self.desired_velocity:.2f} m/s")

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
            # ROS override active
            self.cm.DVA_write(self.q_long_passive, 1)
            self.cm.DVA_write(self.q_gear, 1)  # or handle reverse if needed

            # CRITICAL: Check if desired velocity is ZERO (stop command)
            if abs(self.desired_velocity) < 0.01:  # Desired velocity is zero
                # Apply full brake to stop and hold
                self.cm.DVA_write(self.q_gas, 0.0)
                self.cm.DVA_write(self.q_brake, self.BRAKE_HOLD_VALUE)
                
                if abs(actual_velocity) < self.STOPPED_THRESHOLD:
                    self.get_logger().info(
                        f"🔴 STOPPED & HOLDING: vel={actual_velocity:.3f} m/s, brake={self.BRAKE_HOLD_VALUE}"
                    )
                else:
                    self.get_logger().info(
                        f"🟡 BRAKING TO STOP: vel={actual_velocity:.3f} m/s → 0.0 m/s, brake={self.BRAKE_HOLD_VALUE}"
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
                
                self.get_logger().info(
                    f"🟢 NORMAL CONTROL: target={self.desired_velocity:.2f}, "
                    f"actual={actual_velocity:.2f}, gas={gas:.3f}, brake={brake:.3f}"
                )

        else:
            # Handover to CarMaker / IPGDriver
            self.cm.DVA_write(self.q_long_passive, 0)

            # Set neutral values
            self.cm.DVA_write(self.q_gas,   0.0)
            self.cm.DVA_write(self.q_brake, 0.0)

            # Release ALL active DVA overrides
            self.cm.DVA_release()

            # Logging on transition (prevents spamming)
            if hasattr(self, '_was_in_ros_mode') and self._was_in_ros_mode:
                self.get_logger().info(
                    "⚪ Handover complete: Driver.Long.passive=0 + DVAReleaseQuants sent. "
                    "All VC overrides released → IPGDriver now fully in control."
                )
                self._was_in_ros_mode = False

        self._was_in_ros_mode = in_ros_mode  # track for next cycle

def main(args=None):
    rclpy.init(args=args)
    node = VehicleController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Gentle shutdown – give control back to IPG
        node.cm.DVA_write(node.q_long_passive, 0)
        node.cm.DVA_write(node.q_gas, 0.0)
        node.cm.DVA_write(node.q_brake, 0.0)
        node.cm.DVA_release()
        node.cm.read()

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


# code changed to handle zero desired velocity by applying full brake to stop and hold the vehicle.