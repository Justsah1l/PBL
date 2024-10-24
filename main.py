import cv2
import numpy as np
from time import sleep, time

# Parameters
largura_min = 80  # Minimum rectangle width for vehicle detection
altura_min = 80   # Minimum rectangle height for vehicle detection
offset = 6        # Offset for line crossing detection
pos_linha = 550   # Line position for vehicle counting
delay = 60        # Video frame delay (for simulation)
speed_estimation_interval = 5  # Interval to estimate speed (in frames)

# Detection lists and counters
detec_cam1 = []
detec_cam2 = []
carros_cam1 = 0
carros_cam2 = 0
vehicle_speed_cam1 = []
vehicle_speed_cam2 = []
emergency_detected_cam1 = False
emergency_detected_cam2 = False

# Log file setup
log_file = open("traffic_log.txt", "w")
weather_condition = "rainy"  # Options: "clear", "rainy"
num_lanes = 3
lane_positions = [350, 550, 750]  # Hypothetical lane positions
lane_counts = [0] * num_lanes     # Vehicle counts per lane
lane_speeds = [[] for _ in range(num_lanes)]
# Color detection thresholds (in HSV)
color_ranges = {
    'red': [(0, 100, 100), (10, 255, 255)],
    'blue': [(94, 80, 2), (126, 255, 255)],
    'green': [(40, 40, 40), (70, 255, 255)]
}
vehicle_colors = {'red': 0, 'blue': 0, 'green': 0}

# Function to detect vehicle color
def detect_vehicle_color(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) > 0:
            vehicle_colors[color] += 1
            return color
    return "unknown"

# Weather-adjusted speed
def adjust_speed_for_weather(speed):
    if weather_condition == "rainy":
        return speed * 0.8  # Simulate reduced speed in rain
    return speed

# Calculate traffic density in a given area (lower half of the frame)
def calculate_traffic_densityy(frame, detec):
    frame_height = frame.shape[0]
    vehicles_in_area = sum(1 for (_, y) in detec if y > frame_height / 2)
    density_percentage = (vehicles_in_area / len(detec)) * 100 if detec else 0
    return density_percentage


# Get vehicle center for detection
def get_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Calculate FPS
def calculate_fps(prev_time):
    curr_time = time()
    fps = 1 / (curr_time - prev_time)
    return curr_time, fps

# Estimate vehicle speed (placeholder logic, can be adjusted)
def estimate_speed(x, previous_x):
    speed = abs(x - previous_x) * 0.1  # Basic speed estimation based on x-movement
    return speed

# Classify vehicle type based on bounding box size
def classify_vehicle(w, h):
    area = w * h
    if area < 5000:
        return "Small Vehicle"
    elif area < 15000:
        return "Medium Vehicle"
    else:
        return "Large Vehicle"

# Detect emergency vehicles based on red/blue light color detection
def detect_emergency_vehicle(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red color range (for emergency lights)
    red_lower = np.array([0, 120, 70], np.uint8)
    red_upper = np.array([10, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsv, red_lower, red_upper)

    # Blue color range (for emergency lights)
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([126, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Check if any emergency color is detected
    red_detected = cv2.countNonZero(red_mask)
    blue_detected = cv2.countNonZero(blue_mask)

    if red_detected > 0 or blue_detected > 0:
        return True
    return False

# Centroid tracker for vehicles
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])

        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            distances = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids), axis=2)
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, distances.shape[0])).difference(used_rows)
            unused_cols = set(range(0, distances.shape[1])).difference(used_cols)

            if distances.shape[0] >= distances.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects

# Add tracking to vehicle detection

vehicle_counts = {
    "Car": 0,
    "Truck": 0,
    "Bike": 0
}
def classify_vehicle(w, h):
    """Classify the vehicle based on width and height."""
    print(f"Width: {w}, Height: {h}")  # Log dimensions
    if h > 80 and w > 180:  # Adjusted for trucks
        return "Truck"
    elif h <= 100 and w > 140:  # Adjusted for cars
        return "Car"
    elif h <= 40 and w <= 100: 
        return "Bike"
    return "Unknown"


def calculate_traffic_density(count_cam1, count_cam2):
    # Calculate the difference in vehicle counts
    count_difference = count_cam2 - count_cam1
    # Calculate density as a percentage of the maximum expected count
    max_expected_count = 100  # You can adjust this based on your context
    density_percentage = (count_difference / max_expected_count) * 100
    return density_percentage if density_percentage > 0 else 0  # Avoid negative densities



def classify_lane(x):
    for i, pos in enumerate(lane_positions):
        if x < pos:
            return i
    return len(lane_positions) - 1
    
# Process each frame and detect vehicles
def process_frame(frame, detec, carros, pos_linha, vehicle_speeds, frame_count, emergency_detected):
    global vehicle_counts  # Declare vehicle_counts as global if defined outside
    adjusted_speed = 0.0
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    contorno, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (25, pos_linha), (1200, pos_linha), (176, 130, 39), 2)

    frame_height, frame_width, _ = frame.shape

    for (i, c) in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= largura_min) and (h >= altura_min)
        if not validar_contorno:
            continue

        # Detect vehicle color and type
        vehicle_color = detect_vehicle_color(frame[y:y+h, x:x+w])

        # Draw text for vehicle type and color
        cv2.putText(frame, vehicle_color, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        

        # Draw rectangle and find center
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        centro = get_center(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame, centro, 4, (0, 0, 255), -1)

        # Vehicle speed estimation (every 5 frames)
        if len(detec) > 1 and frame_count % speed_estimation_interval == 0:
            previous_x, previous_y = detec[-2]
            current_speed = estimate_speed(centro[0], previous_x)
            adjusted_speed = adjust_speed_for_weather(current_speed)
            vehicle_speeds.append(adjusted_speed)

        # Check if the vehicle crosses the counting line
        if len(detec) > 1 and frame_count % speed_estimation_interval == 0:
            lane_index = classify_lane(centro[0])
            
            # Use the last detected vehicle's position to check crossing
            if previous_y < pos_linha <= centro[1]:  # Vehicle crosses the line downwards
                lane_counts[lane_index] += 1
                lane_speeds[lane_index].append(adjusted_speed)

                # Log lane, vehicle type, color, speed, and count
                log_file.write(f"Lane: {lane_index + 1}, Color: {vehicle_color}, Speed: {adjusted_speed:.2f}, Count: {lane_counts[lane_index]}\n")

    frame_width = frame.shape[1]
    margin_right = 10  # Adjust this value to move text closer or further from the right edge

    cv2.putText(frame, f"Cars: {vehicle_counts.get('Car', 0)}", (frame_width - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Trucks: {vehicle_counts.get('Truck', 0)}", (frame_width - 150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Bikes: {vehicle_counts.get('Bike', 0)}", (frame_width - 150, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Detect emergency vehicle
    if detect_emergency_vehicle(frame):
        emergency_detected = True
        log_file.write("Emergency vehicle detected!\n")

    # Calculate traffic density
    density = calculate_traffic_density(carros_cam1, carros_cam2)
    cv2.putText(frame, f"Traffic Density: {density:.2f}%", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Check for vehicles crossing the line
    for (x, y) in detec:
        if (y < (pos_linha + offset)) and (y > (pos_linha - offset)):
            vehicle_type = classify_vehicle(w, h)  
            print(f"Detected vehicle type: {vehicle_type}")

        # Increment vehicle count based on type
            if vehicle_type in vehicle_counts:
                vehicle_counts[vehicle_type] += 1

            cv2.putText(frame, vehicle_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            carros += 1
            cv2.line(frame, (25, pos_linha), (1200, pos_linha), (0, 127, 255), 3)
            detec.remove((x, y))
            print(f"Car detected. Current count: {carros}")

    return frame, carros, vehicle_speeds, emergency_detected





# Main loop and other parts remain unchanged
cap1 = cv2.VideoCapture('video.mp4')
cap2 = cv2.VideoCapture('video.mp4')

subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

prev_time = time()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        break

    # FPS Calculation
    prev_time, fps = calculate_fps(prev_time)

    # Process frames for both cameras
    frame_count = int(cap1.get(cv2.CAP_PROP_POS_FRAMES))  # Frame number
    frame1, carros_cam1, vehicle_speed_cam1, emergency_detected_cam1 = process_frame(
        frame1, detec_cam1, carros_cam1, pos_linha, vehicle_speed_cam1, frame_count, emergency_detected_cam1)
    frame2, carros_cam2, vehicle_speed_cam2, emergency_detected_cam2 = process_frame(
        frame2, detec_cam2, carros_cam2, pos_linha, vehicle_speed_cam2, frame_count, emergency_detected_cam2)

    # Traffic Status and Congestion Detection
    if carros_cam1 > carros_cam2 + 10:
        traffic_status = "Possible Traffic Jam or Blockage"
    elif carros_cam2 > carros_cam1:
        traffic_status = "Traffic Flowing Smoothly"
    else:
        traffic_status = "Normal Traffic Flow"

    if len(vehicle_speed_cam1) > 0:
        avg_speed_cam1 = sum(vehicle_speed_cam1) / len(vehicle_speed_cam1)
        cv2.putText(frame1, f"Average Speed: {avg_speed_cam1:.2f} km/h", (50, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    if len(vehicle_speed_cam2) > 0:
        avg_speed_cam2 = sum(vehicle_speed_cam2) / len(vehicle_speed_cam2)
        cv2.putText(frame2, f"Average Speed: {avg_speed_cam2:.2f} km/h", (50, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display vehicle count, traffic status, and emergency detection on frames
    cv2.putText(frame1, f"CAM1 COUNT: {carros_cam1}", (50, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)
    cv2.putText(frame2, f"CAM2 COUNT: {carros_cam2}", (50, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 4)
    cv2.putText(frame1, f"TRAFFIC STATUS: {traffic_status}", (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 4)
    cv2.putText(frame1, f"FPS: {fps:.2f}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Display frames
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    if cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()
cap1.release()
cap2.release()
log_file.close()
