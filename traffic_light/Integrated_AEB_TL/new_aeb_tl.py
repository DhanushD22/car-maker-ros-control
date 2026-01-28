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

# Lane Detection Parameters - Enhanced
ROI_TOP_RATIO = 0.50  # Start ROI from 50% of frame
ROI_BOTTOM_RATIO = 1.0  # Bottom of frame
WHITE_THRESHOLD = 180  # Higher threshold for better white detection
GAUSSIAN_KERNEL = 7  # Larger kernel for better noise reduction
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 40  # Higher threshold for more robust lines
HOUGH_MIN_LINE_LEN = 30
HOUGH_MAX_LINE_GAP = 100
MIN_SLOPE = 0.3  # Minimum slope to consider as lane line

# Traffic Light Parameters
TL_ROI_TOP = 0.0
TL_ROI_BOTTOM = 0.35
TL_MIN_AREA = 35
TL_BLUE_LOWER  = np.array([70,  50,  50])
TL_BLUE_UPPER  = np.array([140, 255, 255])
TL_GREEN_LOWER = np.array([35,  80,  80])
TL_GREEN_UPPER = np.array([85,  255, 255])

# Object Detection Parameters (YOLO-based)
YOLO_MODEL = "yolov8n.pt"  # Nano model for speed
YOLO_CONF_THRESHOLD = 0.4  # Confidence threshold
YOLO_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
EMERGENCY_BRAKE_DISTANCE = 2.0  # meters - full brake
SAFE_FOLLOWING_DISTANCE = 3.0   # meters - maintain this distance
WARNING_DISTANCE = 6.0          # meters - start slowing down
DECEL_DISTANCE = 10.0           # meters - start gentle deceleration for static objects

# Motion detection parameters
MOTION_THRESHOLD = 10  # pixels - movement to consider object as moving
STATIONARY_FRAMES = 3  # frames to confirm object is stationary

# Distance estimation calibration
CAMERA_FOCAL_LENGTH = 800  # pixels
KNOWN_CAR_WIDTH = 1.8      # meters
KNOWN_CAR_HEIGHT = 1.5     # meters

# ACC Control - DISABLED (CarMaker handles it)
ACC_ENABLED = False  # Let CarMaker handle cruise control

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
        
        # Publishers - Only Traffic Light and Object Avoidance (AEB only, no ACC)
        self.traffic_light_pub = self.create_publisher(Twist, '/traffic_light_cmd', 10)
        self.object_avoidance_pub = self.create_publisher(Twist, '/object_avoidance_cmd', 10)
        
        self.get_logger().info('ROS2 publishers initialized:')
        self.get_logger().info('  - /traffic_light_cmd (Traffic Light)')
        self.get_logger().info('  - /object_avoidance_cmd (AEB only - ACC disabled)')
    
    def publish_traffic_light(self, velocity):
        msg = Twist()
        msg.linear.x = velocity
        self.traffic_light_pub.publish(msg)
        self.get_logger().info(f'[TRAFFIC] Published: {velocity:.2f} m/s')
    
    def publish_object_avoidance(self, velocity, brake_force, object_type):
        """Object avoidance command (AEB only)"""
        msg = Twist()
        msg.linear.x = velocity
        msg.angular.z = brake_force  # Brake force
        self.object_avoidance_pub.publish(msg)
        self.get_logger().warning(f'[AEB] {object_type}: vel={velocity:.2f} m/s, brake={brake_force:.2f}')

# ======================== LANE DETECTION - ENHANCED ========================
class LaneDetector:
    def __init__(self):
        self.frame_count = 0
        self.left_lane_history = deque(maxlen=8)
        self.right_lane_history = deque(maxlen=8)
        
    def preprocess_frame(self, frame):
        """Enhanced preprocessing for better lane detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
        
        # White lane detection
        _, white_mask = cv2.threshold(blurred, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Yellow lane detection (in HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_lower = np.array([15, 80, 80])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Combine white and yellow masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Edge detection
        edges = cv2.Canny(combined_mask, CANNY_LOW, CANNY_HIGH)
        
        return edges, combined_mask
    
    def get_roi_mask(self, frame):
        """Create region of interest mask"""
        h, w = frame.shape[:2]
        roi_top = int(h * ROI_TOP_RATIO)
        roi_bottom = int(h * ROI_BOTTOM_RATIO)
        
        # Trapezoid shaped ROI
        vertices = np.array([[
            (int(w * 0.05), roi_bottom),      # Bottom left
            (int(w * 0.45), roi_top),         # Top left
            (int(w * 0.55), roi_top),         # Top right
            (int(w * 0.95), roi_bottom)       # Bottom right
        ]], dtype=np.int32)
        
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, vertices, 255)
        
        return mask, vertices
    
    def detect_lines(self, edges, mask):
        """Detect lines using Hough transform"""
        masked_edges = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(
            masked_edges, 
            rho=1, 
            theta=np.pi/180,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LEN,
            maxLineGap=HOUGH_MAX_LINE_GAP
        )
        return lines, masked_edges
    
    def classify_lines(self, lines, frame_width):
        """Classify lines as left or right lane"""
        left_lines = []
        right_lines = []
        
        if lines is None:
            return left_lines, right_lines
        
        center_x = frame_width / 2
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate slope
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter by slope
            if abs(slope) < MIN_SLOPE:
                continue
            
            # Calculate midpoint
            mid_x = (x1 + x2) / 2
            
            # Classify based on slope and position
            if slope < 0 and mid_x < center_x * 1.3:
                # Left lane (negative slope)
                left_lines.append(line[0])
            elif slope > 0 and mid_x > center_x * 0.7:
                # Right lane (positive slope)
                right_lines.append(line[0])
        
        return left_lines, right_lines
    
    def fit_lane_line(self, lines, frame_height):
        """Fit a line through detected lane segments"""
        if not lines or len(lines) < 2:
            return None
        
        # Collect all points from line segments
        points = []
        for x1, y1, x2, y2 in lines:
            points.extend([(x1, y1), (x2, y2)])
        
        if len(points) < 4:
            return None
        
        points = np.array(points)
        
        # Fit polynomial (degree 1 for straight line)
        try:
            coeffs = np.polyfit(points[:,1], points[:,0], 1)
        except:
            return None
        
        # Calculate line endpoints
        y1 = frame_height
        y2 = int(frame_height * ROI_TOP_RATIO)
        x1 = int(coeffs[0] * y1 + coeffs[1])
        x2 = int(coeffs[0] * y2 + coeffs[1])
        
        return (x1, y1, x2, y2)
    
    def average_lane(self, lane_history):
        """Average lane positions over time for stability"""
        if not lane_history:
            return None
        lanes = np.array(list(lane_history))
        return tuple(np.mean(lanes, axis=0).astype(int))
    
    def detect_lanes(self, frame):
        """Main lane detection pipeline"""
        self.frame_count += 1
        h, w = frame.shape[:2]
        
        # Preprocess
        edges, lane_mask = self.preprocess_frame(frame)
        
        # Get ROI
        roi_mask, roi_vertices = self.get_roi_mask(edges)
        
        # Detect lines
        lines, masked_edges = self.detect_lines(edges, roi_mask)
        
        # Classify lines
        left_lines, right_lines = self.classify_lines(lines, w)
        
        # Fit lane lines
        left_lane = self.fit_lane_line(left_lines, h)
        right_lane = self.fit_lane_line(right_lines, h)
        
        # Update history
        if left_lane:
            self.left_lane_history.append(left_lane)
        if right_lane:
            self.right_lane_history.append(right_lane)
        
        # Get averaged lanes
        left_avg = self.average_lane(self.left_lane_history)
        right_avg = self.average_lane(self.right_lane_history)
        
        # Create visualization
        vis_frame = self.visualize_lanes(frame, left_avg, right_avg, roi_vertices, 
                                        left_lines, right_lines, masked_edges)
        
        return left_avg, right_avg, vis_frame
    
    def visualize_lanes(self, frame, left_lane, right_lane, roi_vertices, 
                       left_lines, right_lines, masked_edges):
        """Create comprehensive lane visualization"""
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # Create overlay for lane area
        lane_overlay = np.zeros_like(vis)
        
        # Draw ROI
        cv2.polylines(vis, [roi_vertices], True, (100, 100, 100), 2)
        
        # Draw detected line segments
        if left_lines:
            for x1, y1, x2, y2 in left_lines:
                cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)
        if right_lines:
            for x1, y1, x2, y2 in right_lines:
                cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        
        # Draw fitted lane lines
        if left_lane:
            x1, y1, x2, y2 = left_lane
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 8)
            cv2.putText(vis, "LEFT", (x2 - 50, y2 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if right_lane:
            x1, y1, x2, y2 = right_lane
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 8)
            cv2.putText(vis, "RIGHT", (x2 - 50, y2 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Fill lane area
        if left_lane and right_lane:
            x1_l, y1_l, x2_l, y2_l = left_lane
            x1_r, y1_r, x2_r, y2_r = right_lane
            
            lane_points = np.array([
                [x1_l, y1_l],
                [x2_l, y2_l],
                [x2_r, y2_r],
                [x1_r, y1_r]
            ], dtype=np.int32)
            
            cv2.fillPoly(lane_overlay, [lane_points], (0, 255, 0))
            cv2.addWeighted(vis, 0.7, lane_overlay, 0.3, 0, vis)
        
        # Add status text
        status = "BOTH LANES" if (left_lane and right_lane) else \
                 "LEFT ONLY" if left_lane else \
                 "RIGHT ONLY" if right_lane else \
                 "NO LANES"
        
        status_color = (0, 255, 0) if (left_lane and right_lane) else (0, 255, 255)
        cv2.putText(vis, f"Lane Status: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return vis

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
        
        area_blue  = cv2.countNonZero(mask_blue)
        area_green = cv2.countNonZero(mask_green)
        
        blue_detected = False
        green_detected = False
        
        # Create visualization
        vis_frame = roi.copy()
        
        if area_blue > TL_MIN_AREA:
            self.state = 'blue'
            blue_detected = True
            logger.warning(f"[TRAFFIC] BLUE light detected → STOP")
            cv2.putText(vis_frame, "BLUE - STOP", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif area_green > TL_MIN_AREA:
            self.state = 'green'
            green_detected = True
            logger.info(f"[TRAFFIC] GREEN light detected → GO")
            cv2.putText(vis_frame, "GREEN - GO", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        else:
            self.state = None
            cv2.putText(vis_frame, "NO SIGNAL", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 3)
        
        # Create debug view with masks
        mask_blue_colored = cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2BGR)
        mask_green_colored = cv2.cvtColor(mask_green, cv2.COLOR_GRAY2BGR)
        
        debug_view = np.hstack([vis_frame, mask_blue_colored, mask_green_colored])
        
        return self.state, blue_detected, green_detected, debug_view

# ======================== UNIFIED OBJECT DETECTION (YOLO) - AEB ONLY ========================
class UnifiedObjectDetector:
    def __init__(self):
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO not available. Install with: pip install ultralytics")
        
        self.model = YOLO(YOLO_MODEL)
        logger.info(f"YOLO model {YOLO_MODEL} loaded successfully")
        logger.info("ACC DISABLED - CarMaker handles cruise control")
        
        # Object tracking
        self.tracked_objects = {}
        self.next_id = 0
        
        # Control state (AEB only)
        self.active = False
        self.target_velocity = 0.0
        self.brake_force = 0.0
        self.object_type = "NONE"
        
    def estimate_distance(self, bbox_width, bbox_height):
        """Estimate distance using bounding box dimensions"""
        dist_from_width = (KNOWN_CAR_WIDTH * CAMERA_FOCAL_LENGTH) / bbox_width if bbox_width > 0 else float('inf')
        dist_from_height = (KNOWN_CAR_HEIGHT * CAMERA_FOCAL_LENGTH) / bbox_height if bbox_height > 0 else float('inf')
        distance = (dist_from_width + dist_from_height) / 2
        return distance
    
    def is_in_ego_lane(self, bbox, frame_width, left_lane, right_lane):
        """Check if object is in ego vehicle's lane"""
        x_center = (bbox[0] + bbox[2]) / 2
        lane_center = frame_width / 2
        lane_width_tolerance = frame_width * 0.3
        
        if abs(x_center - lane_center) < lane_width_tolerance:
            return True
        
        if left_lane and right_lane:
            left_x = left_lane[0]
            right_x = right_lane[0]
            if left_x < x_center < right_x:
                return True
        
        return False
    
    def track_object_motion(self, obj_id, current_center):
        """Track object movement"""
        if obj_id not in self.tracked_objects:
            self.tracked_objects[obj_id] = {
                'positions': deque(maxlen=10),
                'stationary_count': 0,
                'is_moving': False
            }
        
        obj = self.tracked_objects[obj_id]
        obj['positions'].append(current_center)
        
        if len(obj['positions']) < 3:
            return False
        
        positions = np.array(list(obj['positions']))
        variance = np.var(positions, axis=0)
        total_movement = np.sum(variance)
        
        if total_movement < MOTION_THRESHOLD:
            obj['stationary_count'] += 1
        else:
            obj['stationary_count'] = 0
        
        obj['is_moving'] = obj['stationary_count'] < STATIONARY_FRAMES
        return obj['is_moving']
    
    def calculate_aeb_output(self, distance, is_moving):
        """
        Calculate AEB output (Emergency braking only - ACC disabled)
        Only triggers for stationary objects or emergency situations
        """
        
        # EMERGENCY: Object very close (regardless of motion)
        if distance < EMERGENCY_BRAKE_DISTANCE:
            self.object_type = "EMERGENCY"
            return 0.0, 1.0
        
        # STATIONARY object ahead - AEB handles this
        if not is_moving:
            if distance < SAFE_FOLLOWING_DISTANCE:
                self.object_type = "STATIC_CLOSE"
                return 0.0, 1.0
            elif distance < WARNING_DISTANCE:
                self.object_type = "STATIC_WARNING"
                decel_ratio = (distance - SAFE_FOLLOWING_DISTANCE) / (WARNING_DISTANCE - SAFE_FOLLOWING_DISTANCE)
                target_vel = 0.5 * decel_ratio
                brake = 0.5 * (1.0 - decel_ratio)
                return target_vel, brake
            elif distance < DECEL_DISTANCE:
                self.object_type = "STATIC_DECEL"
                decel_ratio = (distance - WARNING_DISTANCE) / (DECEL_DISTANCE - WARNING_DISTANCE)
                target_vel = 1.0 + decel_ratio
                brake = 0.2 * (1.0 - decel_ratio)
                return target_vel, brake
        
        # MOVING objects - Let CarMaker handle (ACC disabled)
        else:
            # Only emergency brake for moving objects if too close
            if distance < EMERGENCY_BRAKE_DISTANCE:
                self.object_type = "MOVING_EMERGENCY"
                return 0.0, 1.0
            else:
                # CarMaker handles everything else
                return None, None
        
        return None, None
    
    def detect_objects(self, frame, left_lane, right_lane):
        """Detect objects using YOLO - AEB only mode"""
        h, w = frame.shape[:2]
        
        # Run YOLO detection
        results = self.model(frame, conf=YOLO_CONF_THRESHOLD, classes=YOLO_CLASSES, verbose=False)
        
        debug_frame = frame.copy()
        
        closest_distance = float('inf')
        closest_obj_id = None
        closest_is_moving = False
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
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
                
                # Track motion
                obj_id = self.next_id
                self.next_id += 1
                is_moving = self.track_object_motion(obj_id, center)
                
                # Find closest object
                if distance < closest_distance:
                    closest_distance = distance
                    closest_obj_id = obj_id
                    closest_is_moving = is_moving
                
                # Visualization
                color = (0, 255, 0) if is_moving else (0, 0, 255)
                cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                
                label = f"{'Moving' if is_moving else 'Static'}: {distance:.1f}m"
                cv2.putText(debug_frame, label, (int(x1), int(y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Add confidence
                cv2.putText(debug_frame, f"Conf: {conf:.2f}", (int(x1), int(y2) + 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Calculate AEB output (only for stationary or emergency)
        if closest_distance < DECEL_DISTANCE:
            target_vel, brake = self.calculate_aeb_output(closest_distance, closest_is_moving)
            
            if target_vel is not None:
                self.active = True
                self.target_velocity = target_vel
                self.brake_force = brake
                
                logger.warning(
                    f"[AEB] {self.object_type}: dist={closest_distance:.1f}m, "
                    f"moving={closest_is_moving}, vel={target_vel:.2f}, brake={brake:.2f}"
                )
            else:
                self.active = False
                self.object_type = "CARMAKER_CONTROL" if closest_is_moving else "CLEAR"
        else:
            self.active = False
            self.object_type = "CLEAR"
        
        # Add status overlay
        status_text = f"AEB: {self.object_type}"
        status_color = (0, 0, 255) if self.active else (0, 255, 0)
        cv2.rectangle(debug_frame, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(debug_frame, status_text, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        
        if self.active:
            cv2.putText(debug_frame, f"Vel: {self.target_velocity:.1f} m/s | Brake: {self.brake_force:.2f}", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        else:
            cv2.putText(debug_frame, "ACC: CarMaker Control", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
        
        return self.active, self.target_velocity, self.brake_force, self.object_type, debug_frame

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
    logger.info("Enhanced Control System: Traffic + AEB (ACC Disabled)")
    logger.info(f"Connecting to {TCP_IP}:{TCP_PORT}...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((TCP_IP, TCP_PORT))
    logger.info("Connected")
    
    lane_detector = LaneDetector()
    tl_detector = TrafficLightDetector()
    object_detector = UnifiedObjectDetector()
    
    # Create 3 windows
    cv2.namedWindow("1. YOLO Object Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("2. Lane Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("3. Traffic Light", cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow("1. YOLO Object Detection", 810, 540)
    cv2.resizeWindow("2. Lane Detection", 810, 540)
    cv2.resizeWindow("3. Traffic Light", 810, 270)
    
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
            left_lane, right_lane, lane_vis = lane_detector.detect_lanes(frame)
            tl_state, blue_detected, green_detected, tl_vis = tl_detector.detect(frame)
            obj_active, obj_vel, obj_brake, obj_type, yolo_vis = object_detector.detect_objects(
                frame, left_lane, right_lane
            )
            
            # Publish to ROS2 - Priority: Traffic > AEB
            if ros2_node:
                try:
                    if blue_detected:
                        ros2_node.publish_traffic_light(0.0)
                    elif obj_active:
                        ros2_node.publish_object_avoidance(obj_vel, obj_brake, obj_type)
                    
                    rclpy.spin_once(ros2_node, timeout_sec=0)
                except Exception as e:
                    logger.error(f"ROS2 publish error: {e}")
            
            # FPS calculation
            fps_counter += 1
            current_time = time.time()
            if current_time - fps_time > 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = current_time
                
                # Add FPS to all windows
                cv2.putText(yolo_vis, f"FPS: {fps}", (width - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(lane_vis, f"FPS: {fps}", (width - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display in 3 separate windows
            cv2.imshow("1. YOLO Object Detection", yolo_vis)
            cv2.imshow("2. Lane Detection", lane_vis)
            cv2.imshow("3. Traffic Light", tl_vis)
            
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

#  changed code in Integrated/aeb_tl_acc.py
