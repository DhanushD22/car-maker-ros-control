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
        error = self.desired_velocity - actual_velocity

        # -------------------------------------------------
        # Decide mode: ROS override only on recent NON-ZERO velocity command
        # -------------------------------------------------
        now = self.get_clock().now()
        in_ros_mode = False
        if self.last_cmd_time is not None:
            elapsed = (now - self.last_cmd_time).nanoseconds / 1e9
            if elapsed < self.CONTROL_TIMEOUT and abs(self.desired_velocity) > 0.01:
                in_ros_mode = True

        if in_ros_mode:
            # ROS takes longitudinal control (speed override)
            self.cm.DVA_write(self.q_long_passive, 1)
            self.cm.DVA_write(self.q_gear, 1)  # Forward gear

            gas = 0.0
            brake = 0.0

            # Hold mode (very close to target)
            if abs(error) < self.ERROR_DEADBAND:
                if self.desired_velocity > 0.5:
                    gas = self.STEADY_GAS
                brake = 0.0
            else:
                if error > 0:
                    gas = (self.KF_GAS * self.desired_velocity) + (self.KP_GAS * error)
                    gas = min(max(gas, 0.0), self.MAX_GAS)
                    brake = 0.0
                else:
                    brake = min(self.KP_BRAKE * (-error), self.MAX_BRAKE)
                    gas = 0.0

            self.cm.DVA_write(self.q_gas, gas)
            self.cm.DVA_write(self.q_brake, brake)

            if abs(error) > 0.3 or self.get_clock().now().nanoseconds % 1_000_000_000 < 200_000_000:
                self.get_logger().info(
                    f"[ROS] v_des={self.desired_velocity:.2f} | v_act={actual_velocity:.2f} | "
                    f"err={error:+.2f} | Gas={gas:.3f} | Brake={brake:.3f}"
                )
        else:
            # IPGDriver full control (default behavior)
            self.cm.DVA_write(self.q_long_passive, 0)
            # Do NOT write gas/brake/gear → IPG decides everything
            # (including its own maneuver speed profile, stopping, etc.)

            if self.last_cmd_time is None or (now - self.last_cmd_time).nanoseconds / 1e9 > 2.0:
                self.get_logger().debug("IPGDriver in full control (no recent /cmd_vel)")

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
        node.cm.read()

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()