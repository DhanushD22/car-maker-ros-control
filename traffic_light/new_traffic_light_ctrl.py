import socket
import numpy as np
import cv2
import re
import time
from collections import deque
import logging

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    ROS2_AVAILABLE = True
    logger_init = logging.getLogger("LaneDetection")
    logger_init.info("ROS2 libraries imported successfully")
except ImportError:
    ROS2_AVAILABLE = False
    logger_init = logging.getLogger("LaneDetection")
    logger_init.warning("ROS2 not available - velocity will not be published")

# ======================== CONFIGURATION ========================
TCP_IP = "192.168.0.162"
TCP_PORT = 2210
HEADER_SIZE = 64

DEBUG_VISUALIZATION = True

ROI_TOP_RATIO = 0.35
WHITE_THRESHOLD = 170
GAUSSIAN_KERNEL = 5
CANNY_LOW = 40
CANNY_HIGH = 120
HOUGH_THRESHOLD = 30
HOUGH_MIN_LINE_LEN = 20
HOUGH_MAX_LINE_GAP = 150

MAX_VELOCITY = 10.0
MIN_VELOCITY = 0.9
SAFE_VELOCITY = 1.0
ACCEL_RATE = 0.5
DECEL_RATE = 1.0
VELOCITY_SMOOTH_WINDOW = 5

MIN_LANE_WIDTH = 100
MAX_LANE_WIDTH = 400
MIN_CONFIDENCE = 0.6

# Traffic Light Parameters
TL_ROI_TOP = 0.0
TL_ROI_BOTTOM = 0.35
TL_MIN_AREA = 35

TL_STOP_Y_THRESH    = 0.25
TL_RESUME_Y_THRESH  = 0.18

TL_BLUE_LOWER  = np.array([70,  50,  50])
TL_BLUE_UPPER  = np.array([140, 255, 255])
TL_GREEN_LOWER = np.array([35,  80,  80])
TL_GREEN_UPPER = np.array([85,  255, 255])

HEADER_REGEX = re.compile(
    rb"\*RSDS\s+(\d+)\s+(\S+)\s+([\d\.]+)\s+(\d+)x(\d+)\s+(\d+)"
)

# ======================== LOGGING SETUP ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("LaneDetection")

# ======================== RSDS FUNCTIONS ========================
def rsds_recv_hdr(sock):
    hdr = bytearray()
    while True:
        while len(hdr) < HEADER_SIZE:
            chunk = sock.recv(HEADER_SIZE - len(hdr))
            if not chunk:
                return None
            hdr.extend(chunk)
        
        if hdr[0] == ord('*') and ord('A') <= hdr[1] <= ord('Z'):
            while hdr and hdr[-1] <= 32:
                hdr.pop()
            return bytes(hdr)
        
        i = 1
        while i < len(hdr) and hdr[i] != ord('*'):
            i += 1
        hdr = hdr[i:]

# ======================== ROS2 PUBLISHER NODE ========================
class VelocityPublisher(Node):
    def __init__(self):
        super().__init__('lane_detection_velocity_publisher')
        self.publisher = self.create_publisher(Twist, '/cmd_vel_command', 10)
        self.get_logger().info('ROS2 /cmd_vel_command publisher initialized')
    
    def publish_velocity(self, linear_velocity):
        msg = Twist()
        msg.linear.x = linear_velocity
        self.publisher.publish(msg)
        self.get_logger().info(f'Published velocity: {linear_velocity:.2f} m/s')

# ======================== LANE DETECTION ========================
class LaneDetector:
    def __init__(self):
        self.frame_count = 0
        self.left_lane_history = deque(maxlen=5)
        self.right_lane_history = deque(maxlen=5)
        
    def preprocess_frame(self, frame):
        logger.debug(f"Frame {self.frame_count}: Preprocessing started")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
        _, white_mask = cv2.threshold(blurred, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(white_mask, CANNY_LOW, CANNY_HIGH)
        logger.debug(f"Frame {self.frame_count}: Edge pixels detected: {np.count_nonzero(edges)}")
        return edges
    
    def get_roi_mask(self, frame):
        h, w = frame.shape[:2]
        roi_top = int(h * ROI_TOP_RATIO)
        vertices = np.array([[
            (0, h),
            (w * 0.35, roi_top),
            (w * 0.65, roi_top),
            (w, h)
        ]], dtype=np.int32)
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, vertices, 255)
        return mask
    
    def detect_lines(self, edges, mask):
        masked_edges = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(
            masked_edges, rho=1, theta=np.pi/180,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LEN,
            maxLineGap=HOUGH_MAX_LINE_GAP
        )
        if lines is not None:
            logger.debug(f"Frame {self.frame_count}: Detected {len(lines)} line segments")
        else:
            logger.warning(f"Frame {self.frame_count}: No lines detected")
        return lines
    
    def classify_lines(self, lines, frame_width):
        left_lines = []
        right_lines = []
        if lines is None:
            return left_lines, right_lines
        center_x = frame_width / 2
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.2:
                continue
            mid_x = (x1 + x2) / 2
            if slope < 0 and mid_x < center_x * 1.2:
                left_lines.append(line[0])
            elif slope > 0 and mid_x > center_x * 0.8:
                right_lines.append(line[0])
        return left_lines, right_lines
    
    def fit_lane_line(self, lines, frame_height):
        if not lines:
            return None
        points = []
        for x1, y1, x2, y2 in lines:
            points.extend([(x1, y1), (x2, y2)])
        if len(points) < 2:
            return None
        points = np.array(points)
        coeffs = np.polyfit(points[:,1], points[:,0], 1)
        y1 = frame_height
        y2 = int(frame_height * ROI_TOP_RATIO)
        x1 = int(coeffs[0] * y1 + coeffs[1])
        x2 = int(coeffs[0] * y2 + coeffs[1])
        return (x1, y1, x2, y2)
    
    def detect_lanes(self, frame):
        self.frame_count += 1
        h, w = frame.shape[:2]
        edges = self.preprocess_frame(frame)
        roi_mask = self.get_roi_mask(edges)
        lines = self.detect_lines(edges, roi_mask)
        left_lines, right_lines = self.classify_lines(lines, w)
        
        if DEBUG_VISUALIZATION and lines is not None:
            debug_frame = frame.copy()
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(debug_frame, (x1,y1), (x2,y2), (0,255,255), 1)
            for ln in left_lines:
                cv2.line(debug_frame, (ln[0],ln[1]), (ln[2],ln[3]), (0,255,0), 2)
            for ln in right_lines:
                cv2.line(debug_frame, (ln[0],ln[1]), (ln[2],ln[3]), (0,0,255), 2)
            
            debug_frame_resized = cv2.resize(debug_frame, (810, 540))
            edges_resized = cv2.resize(edges, (810, 540))
            
            cv2.namedWindow("Debug Lines", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Debug Edges", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Debug Lines", 810, 540)
            cv2.resizeWindow("Debug Edges", 810, 540)
            
            cv2.imshow("Debug Lines", debug_frame_resized)
            cv2.imshow("Debug Edges", edges_resized)
        
        left_lane = self.fit_lane_line(left_lines, h)
        right_lane = self.fit_lane_line(right_lines, h)
        
        if left_lane: self.left_lane_history.append(left_lane)
        if right_lane: self.right_lane_history.append(right_lane)
        
        left_avg = self.average_lane(self.left_lane_history) if self.left_lane_history else None
        right_avg = self.average_lane(self.right_lane_history) if self.right_lane_history else None
        
        logger.info(f"Frame {self.frame_count}: Left lane: {left_avg is not None}, Right lane: {right_avg is not None}")
        return left_avg, right_avg
    
    def average_lane(self, lane_history):
        if not lane_history:
            return None
        lanes = np.array(list(lane_history))
        return tuple(np.mean(lanes, axis=0).astype(int))

# ======================== TRAFFIC LIGHT DETECTION ========================
class TrafficLightDetector:
    def __init__(self):
        self.state = None

    def detect(self, frame):
        h, w = frame.shape[:2]
        roi = frame[0:int(h * TL_ROI_BOTTOM), :]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        mask_blue  = cv2.inRange(hsv, TL_BLUE_LOWER,  TL_BLUE_UPPER)
        mask_green = cv2.inRange(hsv, TL_GREEN_LOWER, TL_GREEN_UPPER)
        
        if DEBUG_VISUALIZATION:
            mask_blue_resized = cv2.resize(mask_blue, (810, 540))
            mask_green_resized = cv2.resize(mask_green, (810, 540))
            
            cv2.namedWindow("Blue Mask", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Green Mask", cv2.WINDOW_NORMAL)
            
            cv2.resizeWindow("Blue Mask", 810, 540)
            cv2.resizeWindow("Green Mask", 810, 540)
            
            cv2.imshow("Blue Mask", mask_blue_resized)
            cv2.imshow("Green Mask", mask_green_resized)
            cv2.waitKey(1)

        area_blue  = cv2.countNonZero(mask_blue)
        area_green = cv2.countNonZero(mask_green)
        
        blue_detected = False
        green_detected = False
        
        # Priority: Blue overrides Green (safety first)
        if area_blue > TL_MIN_AREA:
            self.state = 'blue'
            blue_detected = True
            logger.warning(f"Traffic light: BLUE detected (area:{area_blue}) → ROS2 STOP")
        elif area_green > TL_MIN_AREA:
            self.state = 'green'
            green_detected = True
            logger.info(f"Traffic light: GREEN detected (area:{area_green}) → CarMaker control")
        else:
            self.state = None
            logger.info(f"Traffic light: NONE (blue:{area_blue}, green:{area_green}) → CarMaker control")
        
        return self.state, blue_detected, green_detected

# ======================== VISUALIZATION ========================
def draw_lanes(frame, left_lane, right_lane, is_safe):
    overlay = frame.copy()
    
    if left_lane and right_lane:
        h, w = frame.shape[:2]
        
        max_fill_y = int(h * 0.48)
        bottom_y = h
        
        pts = np.array([
            [left_lane[0],  bottom_y],
            [left_lane[2],  max_fill_y],
            [right_lane[2], max_fill_y],
            [right_lane[0], bottom_y]
        ], dtype=np.int32)
        
        area = cv2.contourArea(pts)
        if area > 8000:
            color = (0, 220, 0) if is_safe else (0, 80, 220)
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        else:
            logger.debug(f"Skipped fill — area too small: {area}")
    
    return frame

def draw_status_hud(frame, tl_state, blue_detected, control_mode):
    h, w = frame.shape[:2]
    
    box_w = 160
    box_h = 70
    cv2.rectangle(frame, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    
    # Color based on control mode
    if control_mode == "ROS2_STOP":
        status_color = (0, 0, 255)  # Red
        status_text = "ROS2 STOP"
    elif control_mode == "CARMAKER":
        status_color = (0, 255, 0)  # Green
        status_text = "CarMaker GO"
    else:
        status_color = (128, 128, 128)  # Gray
        status_text = "IDLE"
    
    cv2.rectangle(frame, (8, 8), (8 + box_w, 8 + box_h), status_color, 2)
    cv2.putText(frame, status_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
    cv2.putText(frame, f"TL: {tl_state or '—'}", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,255), 1)
    
    # Publishing status
    pub_status = "Publishing: YES" if blue_detected else "Publishing: NO"
    pub_color = (0, 255, 255) if blue_detected else (100, 100, 100)
    cv2.putText(frame, pub_status, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.35, pub_color, 1)
    
    return frame

# ======================== MAIN LOOP ========================
def main():
    ros2_node = None
    if ROS2_AVAILABLE:
        try:
            rclpy.init()
            ros2_node = VelocityPublisher()
            logger.info("ROS2 node initialized")
        except Exception as e:
            logger.error(f"ROS2 init failed: {e}")
    
    logger.info("=" * 60)
    logger.info("Lane + Traffic Light Detection & Control")
    logger.info("BLUE LIGHT → ROS2 stops vehicle (publishes 0.0 m/s)")
    logger.info("GREEN LIGHT → CarMaker control (no publishing)")
    logger.info("NO LIGHT → CarMaker control (no publishing)")
    logger.info(f"Connecting to {TCP_IP}:{TCP_PORT}...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((TCP_IP, TCP_PORT))
    logger.info("Connected")
    
    lane_detector = LaneDetector()
    tl_detector = TrafficLightDetector()
    
    cv2.namedWindow("Lane Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Lane Detection", 810, 540)
    
    fps_time = time.time()
    fps_counter = 0
    
    try:
        while True:
            hdr = rsds_recv_hdr(sock)
            if hdr is None:
                logger.error("Connection lost")
                break
            
            match = HEADER_REGEX.search(hdr)
            if not match:
                continue
            
            width = int(match.group(4))
            height = int(match.group(5))
            img_len = int(match.group(6))
            
            raw = b""
            while len(raw) < img_len:
                pkt = sock.recv(img_len - len(raw))
                if not pkt:
                    return
                raw += pkt
            
            if img_len != width * height * 3:
                continue
            
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
            
            left_lane, right_lane = lane_detector.detect_lanes(frame)
            
            # Traffic light detection
            tl_state, blue_detected, green_detected = tl_detector.detect(frame)
            
            # Determine control mode and velocity
            if blue_detected:
                # BLUE LIGHT: ROS2 takes control and STOPS the vehicle
                control_mode = "ROS2_STOP"
                velocity = 0.0
                should_publish = True
                logger.warning(" BLUE LIGHT → ROS2 publishing 0.0 m/s (STOP)")
            elif green_detected:
                # GREEN LIGHT: Give control to CarMaker (don't publish)
                control_mode = "CARMAKER"
                velocity = 0.0  # Not used, just for display
                should_publish = False
                logger.info(" GREEN LIGHT → No publishing (CarMaker control)")
            else:
                # NO LIGHT: Give control to CarMaker (don't publish)
                control_mode = "CARMAKER"
                velocity = 0.0  # Not used, just for display
                should_publish = False
                logger.info(" NO LIGHT → No publishing (CarMaker control)")
            
            # ROS2 publish logic
            if ros2_node and should_publish:
                try:
                    ros2_node.publish_velocity(velocity)
                    rclpy.spin_once(ros2_node, timeout_sec=0)
                except Exception as e:
                    logger.error(f"ROS2 publish error: {e}")
            
            # Visualization
            is_safe = (left_lane is not None and right_lane is not None)
            output_frame = draw_lanes(frame, left_lane, right_lane, is_safe)
            output_frame = draw_status_hud(output_frame, tl_state, blue_detected, control_mode)
            
            # FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_time > 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = current_time
                cv2.putText(output_frame, f"FPS: {fps}", (10, height - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("Lane Detection", output_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                logger.info("Shutdown requested")
                break
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        sock.close()
        cv2.destroyAllWindows()
        if ros2_node:
            try:
                ros2_node.destroy_node()
                rclpy.shutdown()
            except:
                pass
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()



# publishing to new topic /cmd_vel_command 
# vehicle_control.py also subscribes to the same topic but publishes to /cmd_vel
