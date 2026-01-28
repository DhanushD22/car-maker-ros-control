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
    from std_msgs.msg import Float32
    ROS2_AVAILABLE = True
    logger_init = logging.getLogger("IntegratedControl")
    logger_init.info("ROS2 libraries imported successfully")
except ImportError:
    ROS2_AVAILABLE = False
    logger_init = logging.getLogger("IntegratedControl")
    logger_init.warning("ROS2 not available")

# YOLO imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger_init.info("YOLO imported successfully")
except ImportError:
    YOLO_AVAILABLE = False
    logger_init.error("YOLO not available - please install: pip install ultralytics")

# ======================== CONFIGURATION ========================
TCP_IP = "192.168.0.162"
TCP_PORT = 2210
HEADER_SIZE = 64

DEBUG_VISUALIZATION = True

# Lane Detection Parameters
ROI_TOP_RATIO = 0.35
WHITE_THRESHOLD = 170
GAUSSIAN_KERNEL = 5
CANNY_LOW = 40
CANNY_HIGH = 120
HOUGH_THRESHOLD = 30
HOUGH_MIN_LINE_LEN = 20
HOUGH_MAX_LINE_GAP = 150

# Traffic Light Parameters
TL_ROI_TOP = 0.0
TL_ROI_BOTTOM = 0.35
TL_MIN_AREA = 35
TL_BLUE_LOWER  = np.array([70,  50,  50])
TL_BLUE_UPPER  = np.array([140, 255, 255])
TL_GREEN_LOWER = np.array([35,  80,  80])
TL_GREEN_UPPER = np.array([85,  255, 255])

# Object Detection Parameters (YOLO-based)
YOLO_MODEL = "yolov8n.pt"  # Nano model for speed (can use yolov8s.pt for better accuracy)
YOLO_CONF_THRESHOLD = 0.4  # Confidence threshold
YOLO_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
EMERGENCY_BRAKE_DISTANCE = 2.0  # meters - full brake
SAFE_FOLLOWING_DISTANCE = 3.0   # meters - maintain this distance from moving objects
WARNING_DISTANCE = 6.0          # meters - start slowing down
DECEL_DISTANCE = 10.0           # meters - start gentle deceleration for static objects

# Motion detection parameters
MOTION_THRESHOLD = 10  # pixels - movement to consider object as moving
STATIONARY_FRAMES = 3  # frames to confirm object is stationary

# Distance estimation calibration
CAMERA_FOCAL_LENGTH = 800  # pixels
KNOWN_CAR_WIDTH = 1.8      # meters
KNOWN_CAR_HEIGHT = 1.5     # meters

HEADER_REGEX = re.compile(
    rb"\*RSDS\s+(\d+)\s+(\S+)\s+([\d\.]+)\s+(\d+)x(\d+)\s+(\d+)"
)

# ======================== LOGGING SETUP ========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("IntegratedControl")

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
class IntegratedControlPublisher(Node):
    def __init__(self):
        super().__init__('integrated_control_publisher')
        
        # Single unified command topic for object avoidance (combines AEB + ACC)
        self.traffic_light_pub = self.create_publisher(Twist, '/traffic_light_cmd', 10)
        self.object_avoidance_pub = self.create_publisher(Twist, '/object_avoidance_cmd', 10)
        
        self.get_logger().info('ROS2 publishers initialized:')
        self.get_logger().info('  - /traffic_light_cmd')
        self.get_logger().info('  - /object_avoidance_cmd (unified AEB + ACC)')
    
    def publish_traffic_light(self, velocity):
        msg = Twist()
        msg.linear.x = velocity
        self.traffic_light_pub.publish(msg)
        self.get_logger().info(f'[TRAFFIC] Published: {velocity:.2f} m/s')
    
    def publish_object_avoidance(self, velocity, brake_force, object_type):
        """Unified object avoidance command"""
        msg = Twist()
        msg.linear.x = velocity
        msg.angular.z = brake_force  # Brake force
        self.object_avoidance_pub.publish(msg)
        self.get_logger().info(f'[OBJ_AVOID] {object_type}: vel={velocity:.2f} m/s, brake={brake_force:.2f}')

# ======================== LANE DETECTION ========================
class LaneDetector:
    def __init__(self):
        self.frame_count = 0
        self.left_lane_history = deque(maxlen=5)
        self.right_lane_history = deque(maxlen=5)
        
    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
        _, white_mask = cv2.threshold(blurred, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(white_mask, CANNY_LOW, CANNY_HIGH)
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
        
        left_lane = self.fit_lane_line(left_lines, h)
        right_lane = self.fit_lane_line(right_lines, h)
        
        if left_lane: self.left_lane_history.append(left_lane)
        if right_lane: self.right_lane_history.append(right_lane)
        
        left_avg = self.average_lane(self.left_lane_history) if self.left_lane_history else None
        right_avg = self.average_lane(self.right_lane_history) if self.right_lane_history else None
        
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
            mask_blue_resized = cv2.resize(mask_blue, (405, 270))
            mask_green_resized = cv2.resize(mask_green, (405, 270))
            
            cv2.namedWindow("TL: Blue", cv2.WINDOW_NORMAL)
            cv2.namedWindow("TL: Green", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("TL: Blue", 405, 270)
            cv2.resizeWindow("TL: Green", 405, 270)
            
            cv2.imshow("TL: Blue", mask_blue_resized)
            cv2.imshow("TL: Green", mask_green_resized)

        area_blue  = cv2.countNonZero(mask_blue)
        area_green = cv2.countNonZero(mask_green)
        
        blue_detected = False
        green_detected = False
        
        if area_blue > TL_MIN_AREA:
            self.state = 'blue'
            blue_detected = True
            logger.warning(f"[TRAFFIC] BLUE light detected → STOP")
        elif area_green > TL_MIN_AREA:
            self.state = 'green'
            green_detected = True
            logger.info(f"[TRAFFIC] GREEN light detected → GO")
        else:
            self.state = None
        
        return self.state, blue_detected, green_detected

# ======================== UNIFIED OBJECT DETECTION (YOLO) ========================
class UnifiedObjectDetector:
    def __init__(self):
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO not available. Install with: pip install ultralytics")
        
        self.model = YOLO(YOLO_MODEL)
        logger.info(f"YOLO model {YOLO_MODEL} loaded successfully")
        
        # Object tracking
        self.tracked_objects = {}  # {id: {'bbox': [], 'positions': deque, 'is_moving': bool}}
        self.next_id = 0
        
        # Control state
        self.active = False
        self.closest_object = None
        self.target_velocity = 0.0
        self.brake_force = 0.0
        self.object_type = "NONE"
        
    def estimate_distance(self, bbox_width, bbox_height):
        """Estimate distance using bounding box dimensions"""
        # Use average of width and height estimation
        dist_from_width = (KNOWN_CAR_WIDTH * CAMERA_FOCAL_LENGTH) / bbox_width if bbox_width > 0 else float('inf')
        dist_from_height = (KNOWN_CAR_HEIGHT * CAMERA_FOCAL_LENGTH) / bbox_height if bbox_height > 0 else float('inf')
        
        # Take the average for better accuracy
        distance = (dist_from_width + dist_from_height) / 2
        return distance
    
    def is_in_ego_lane(self, bbox, frame_width, left_lane, right_lane):
        """Check if object is in ego vehicle's lane"""
        x_center = (bbox[0] + bbox[2]) / 2
        
        # Simple center-based check
        lane_center = frame_width / 2
        lane_width_tolerance = frame_width * 0.3  # 30% of frame width
        
        # Check if object is in center region
        if abs(x_center - lane_center) < lane_width_tolerance:
            return True
        
        # If we have lane lines, use them for better accuracy
        if left_lane and right_lane:
            # Use bottom of bbox (closest point)
            left_x = left_lane[0]
            right_x = right_lane[0]
            
            # Check if object center is between lanes
            if left_x < x_center < right_x:
                return True
        
        return False
    
    def track_object_motion(self, obj_id, current_center):
        """Track object movement to determine if moving or stationary"""
        if obj_id not in self.tracked_objects:
            self.tracked_objects[obj_id] = {
                'positions': deque(maxlen=10),
                'stationary_count': 0,
                'is_moving': False
            }
        
        obj = self.tracked_objects[obj_id]
        obj['positions'].append(current_center)
        
        # Need at least 3 positions to determine motion
        if len(obj['positions']) < 3:
            return False  # Assume stationary initially
        
        # Calculate movement variance
        positions = np.array(list(obj['positions']))
        variance = np.var(positions, axis=0)
        total_movement = np.sum(variance)
        
        # Determine if moving
        if total_movement < MOTION_THRESHOLD:
            obj['stationary_count'] += 1
        else:
            obj['stationary_count'] = 0
        
        # Confirm stationary after consecutive stationary frames
        obj['is_moving'] = obj['stationary_count'] < STATIONARY_FRAMES
        
        return obj['is_moving']
    
    def calculate_control_output(self, distance, is_moving, object_velocity=0.0):
        """
        Calculate target velocity and brake force based on:
        - Distance to object
        - Whether object is moving or stationary
        - Object velocity (if moving)
        """
        
        # EMERGENCY: Object very close
        if distance < EMERGENCY_BRAKE_DISTANCE:
            self.object_type = "EMERGENCY"
            return 0.0, 1.0  # Full stop, full brake
        
        # STATIONARY object ahead
        if not is_moving:
            if distance < SAFE_FOLLOWING_DISTANCE:
                # Too close to stationary - full stop
                self.object_type = "STATIC_CLOSE"
                return 0.0, 1.0
            elif distance < WARNING_DISTANCE:
                # Warning zone - gradual deceleration
                self.object_type = "STATIC_WARNING"
                decel_ratio = (distance - SAFE_FOLLOWING_DISTANCE) / (WARNING_DISTANCE - SAFE_FOLLOWING_DISTANCE)
                target_vel = 0.5 * decel_ratio  # Slow down gradually
                brake = 0.5 * (1.0 - decel_ratio)
                return target_vel, brake
            elif distance < DECEL_DISTANCE:
                # Start slowing down gently
                self.object_type = "STATIC_DECEL"
                decel_ratio = (distance - WARNING_DISTANCE) / (DECEL_DISTANCE - WARNING_DISTANCE)
                target_vel = 1.0 + decel_ratio  # 1.0 to 2.0 m/s
                brake = 0.2 * (1.0 - decel_ratio)
                return target_vel, brake
            else:
                # Far enough - clear
                return None, None
        
        # MOVING object ahead (ACC mode)
        else:
            if distance < SAFE_FOLLOWING_DISTANCE:
                # Too close - match speed or slow down
                self.object_type = "MOVING_CLOSE"
                # Estimate object velocity (simplified - in reality use tracking)
                target_vel = max(0.5, object_velocity * 0.8)  # Match but slightly slower
                brake = 0.6
                return target_vel, brake
            elif distance < WARNING_DISTANCE:
                # Maintain safe distance
                self.object_type = "MOVING_FOLLOW"
                # Proportional velocity based on distance
                distance_ratio = (distance - SAFE_FOLLOWING_DISTANCE) / (WARNING_DISTANCE - SAFE_FOLLOWING_DISTANCE)
                target_vel = 1.0 + distance_ratio  # 1.0 to 2.0 m/s
                brake = 0.3 * (1.0 - distance_ratio)
                return target_vel, brake
            else:
                # Far enough - clear
                return None, None
    
    def detect_objects(self, frame, left_lane, right_lane):
        """Detect objects using YOLO and determine control output"""
        h, w = frame.shape[:2]
        
        # Run YOLO detection
        results = self.model(frame, conf=YOLO_CONF_THRESHOLD, classes=YOLO_CLASSES, verbose=False)
        
        # Process detections
        closest_distance = float('inf')
        closest_obj_id = None
        closest_is_moving = False
        
        if DEBUG_VISUALIZATION:
            debug_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bbox coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                # Check if in ego lane
                if not self.is_in_ego_lane([x1, y1, x2, y2], w, left_lane, right_lane):
                    continue
                
                # Estimate distance
                distance = self.estimate_distance(bbox_width, bbox_height)
                
                # Track object motion
                obj_id = self.next_id
                self.next_id += 1
                is_moving = self.track_object_motion(obj_id, center)
                
                # Find closest object in lane
                if distance < closest_distance:
                    closest_distance = distance
                    closest_obj_id = obj_id
                    closest_is_moving = is_moving
                
                # Visualization
                if DEBUG_VISUALIZATION:
                    color = (0, 255, 0) if is_moving else (0, 0, 255)
                    cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{'Moving' if is_moving else 'Static'} {distance:.1f}m"
                    cv2.putText(debug_frame, label, (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate control output
        if closest_distance < DECEL_DISTANCE:
            target_vel, brake = self.calculate_control_output(closest_distance, closest_is_moving)
            
            if target_vel is not None:  # Object requires action
                self.active = True
                self.closest_object = {
                    'distance': closest_distance,
                    'is_moving': closest_is_moving
                }
                self.target_velocity = target_vel
                self.brake_force = brake
                
                logger.warning(
                    f"[OBJ_DETECT] {self.object_type}: dist={closest_distance:.1f}m, "
                    f"moving={closest_is_moving}, vel={target_vel:.2f}, brake={brake:.2f}"
                )
            else:
                self.active = False
        else:
            self.active = False
            self.object_type = "CLEAR"
        
        # Visualization
        if DEBUG_VISUALIZATION:
            debug_resized = cv2.resize(debug_frame, (810, 540))
            cv2.namedWindow("Object Detection (YOLO)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Object Detection (YOLO)", 810, 540)
            
            status_text = f"Status: {self.object_type}"
            status_color = (0, 0, 255) if self.active else (0, 255, 0)
            cv2.putText(debug_resized, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            cv2.imshow("Object Detection (YOLO)", debug_resized)
        
        return self.active, self.target_velocity, self.brake_force, self.object_type

# ======================== VISUALIZATION ========================
def draw_status_hud(frame, tl_state, obj_active, obj_type, obj_vel, obj_brake):
    h, w = frame.shape[:2]
    
    y_offset = 10
    
    # Traffic Light Status
    tl_color = (0, 0, 255) if tl_state == 'blue' else (0, 255, 0) if tl_state == 'green' else (128, 128, 128)
    tl_text = f"Traffic: {tl_state.upper() if tl_state else 'NONE'}"
    cv2.putText(frame, tl_text, (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tl_color, 2)
    
    # Object Detection Status
    y_offset += 30
    obj_color = (0, 0, 255) if obj_active else (0, 255, 0)
    obj_text = f"Object: {obj_type}"
    if obj_active:
        obj_text += f" (vel:{obj_vel:.1f}, brake:{obj_brake:.2f})"
    cv2.putText(frame, obj_text, (10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, obj_color, 2)
    
    return frame

# ======================== MAIN LOOP ========================
def main():
    if not YOLO_AVAILABLE:
        logger.error("YOLO not available. Install with: pip install ultralytics")
        return
    
    ros2_node = None
    if ROS2_AVAILABLE:
        try:
            rclpy.init()
            ros2_node = IntegratedControlPublisher()
            logger.info("ROS2 node initialized")
        except Exception as e:
            logger.error(f"ROS2 init failed: {e}")
    
    logger.info("=" * 60)
    logger.info("Enhanced Control System: Traffic + Unified Object Detection (YOLO)")
    logger.info(f"Connecting to {TCP_IP}:{TCP_PORT}...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((TCP_IP, TCP_PORT))
    logger.info("Connected")
    
    lane_detector = LaneDetector()
    tl_detector = TrafficLightDetector()
    object_detector = UnifiedObjectDetector()
    
    cv2.namedWindow("Integrated Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Integrated Control", 810, 540)
    
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
            
            # Run all detections
            left_lane, right_lane = lane_detector.detect_lanes(frame)
            tl_state, blue_detected, green_detected = tl_detector.detect(frame)
            obj_active, obj_vel, obj_brake, obj_type = object_detector.detect_objects(frame, left_lane, right_lane)
            
            # Publish to ROS2 with priority: Traffic > Object Avoidance
            if ros2_node:
                try:
                    if blue_detected:
                        # Priority 1: Traffic light
                        ros2_node.publish_traffic_light(0.0)
                    elif obj_active:
                        # Priority 2: Object avoidance (unified AEB + ACC)
                        ros2_node.publish_object_avoidance(obj_vel, obj_brake, obj_type)
                    
                    rclpy.spin_once(ros2_node, timeout_sec=0)
                except Exception as e:
                    logger.error(f"ROS2 publish error: {e}")
            
            # Visualization
            output_frame = draw_status_hud(frame, tl_state, obj_active, obj_type, obj_vel, obj_brake)
            
            # FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_time > 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = current_time
                cv2.putText(output_frame, f"FPS: {fps}", (10, height - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("Integrated Control", output_frame)
            
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



#  changed object detection to - yolov8n.pt for speed and unified AEB + ACC into single object avoidance publisher 