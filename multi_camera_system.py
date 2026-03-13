import cv2
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import os

from config_multicam import *
from src.yolo_detector_improved import YOLODetector
from src.face_recognizer_wrapper import face_recognizer
from src.database import db

@dataclass
class CameraStream:
    index: int
    name: str
    purpose: str  # 'entry', 'exit', 'both'
    cap: cv2.VideoCapture
    frame: Optional[np.ndarray]
    last_update: datetime
    is_active: bool

@dataclass
class CaptureFrame:
    """Represents a frame candidate for capture"""
    frame: np.ndarray
    bbox: Tuple[int, int, int, int]
    confidence: float
    quality_score: float
    face_encoding: Optional[np.ndarray]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class DetectedPerson:
    track_id: int
    camera_source: str
    person_id: Optional[int]
    person_name: str
    confidence: float
    face_encoding: Optional[np.ndarray]
    last_seen: datetime
    detection_count: int
    auto_captured: bool
    capture_candidates: List[CaptureFrame] = None  # Store frame candidates for quality assessment

    def __post_init__(self):
        if self.capture_candidates is None:
            self.capture_candidates = []

class FaceDeduplicationSystem:
    """System to automatically capture and deduplicate faces"""

    def __init__(self):
        self.unknown_faces = {}  # temp_id -> face_encoding
        self.capture_cooldowns = {}  # face_hash -> last_capture_time
        self.temp_person_counter = 0

        # Create auto-capture directory
        AUTO_CAPTURED_PATH.mkdir(parents=True, exist_ok=True)

    def get_face_hash(self, face_encoding: np.ndarray) -> str:
        """Generate hash for face encoding to track cooldowns"""
        face_str = ','.join([f"{x:.4f}" for x in face_encoding[:20]])  # Use first 20 values
        return hashlib.md5(face_str.encode()).hexdigest()

    def calculate_image_quality(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate quality score for a face region in frame"""
        x1, y1, x2, y2 = bbox

        # Extract face region with padding
        padding = 10
        face_x1 = max(0, x1 - padding)
        face_y1 = max(0, y1 - padding)
        face_x2 = min(frame.shape[1], x2 + padding)
        face_y2 = min(frame.shape[0], y2 + padding)

        face_region = frame[face_y1:face_y2, face_x1:face_x2]

        if face_region.size == 0:
            return 0.0

        # Convert to grayscale for quality analysis
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region

        # 1. Sharpness score (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        # 2. Brightness score (avoid too dark/bright)
        brightness = gray.mean()
        brightness_score = 1.0 - abs(brightness - 128) / 128  # Optimal around 128

        # 3. Contrast score (standard deviation)
        contrast = gray.std()
        contrast_score = min(contrast / 64.0, 1.0)  # Normalize to 0-1

        # 4. Face size score (larger faces are better)
        face_area = (x2 - x1) * (y2 - y1)
        size_score = min(face_area / (AUTO_CAPTURE_FACE_SIZE_MIN * AUTO_CAPTURE_FACE_SIZE_MIN), 1.0)

        # 5. Focus score using gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        focus_score = min(gradient_magnitude.mean() / 50.0, 1.0)

        # Weighted combination of quality metrics
        quality_score = (
            0.3 * min(sharpness / 100.0, 1.0) +  # Sharpness (30%)
            0.2 * brightness_score +              # Brightness (20%)
            0.2 * contrast_score +                # Contrast (20%)
            0.15 * size_score +                   # Size (15%)
            0.15 * focus_score                    # Focus (15%)
        )

        return min(quality_score, 1.0)

    def add_capture_candidate(self, person: DetectedPerson, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                            confidence: float, face_encoding: Optional[np.ndarray]):
        """Add a frame as capture candidate and maintain best 10 frames"""
        quality_score = self.calculate_image_quality(frame, bbox)

        # Only consider frames above minimum quality
        if quality_score < 0.3:  # Minimum quality threshold
            return

        candidate = CaptureFrame(
            frame=frame.copy(),
            bbox=bbox,
            confidence=confidence,
            quality_score=quality_score,
            face_encoding=face_encoding
        )

        person.capture_candidates.append(candidate)

        # Keep only best 10 candidates, sorted by quality
        person.capture_candidates.sort(key=lambda x: x.quality_score, reverse=True)
        if len(person.capture_candidates) > 10:
            person.capture_candidates = person.capture_candidates[:10]

    def is_similar_to_known(self, face_encoding: np.ndarray) -> Tuple[bool, Optional[int], float]:
        """Check if face is similar to any known person using enhanced similarity metrics"""
        if not face_recognizer.known_face_encodings:
            return False, None, 0.0

        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Enhanced similarity calculation with multiple metrics
        similarities = []
        euclidean_distances = []

        for known_enc in face_recognizer.known_face_encodings:
            # Cosine similarity (primary metric)
            cos_sim = cosine_similarity([face_encoding], [known_enc])[0][0]
            similarities.append(cos_sim)

            # Euclidean distance (secondary metric)
            eucl_dist = np.linalg.norm(face_encoding - known_enc)
            euclidean_distances.append(eucl_dist)

        # Combined scoring system (weighted average)
        max_cos_similarity = max(similarities)
        min_eucl_distance = min(euclidean_distances)

        # Normalize euclidean distance to 0-1 range (inverse for similarity)
        normalized_eucl = 1 / (1 + min_eucl_distance)

        # Weighted combined score (70% cosine, 30% normalized euclidean)
        combined_score = 0.7 * max_cos_similarity + 0.3 * normalized_eucl

        # Use stricter threshold for better accuracy
        enhanced_threshold = FACE_SIMILARITY_THRESHOLD + 0.1

        if combined_score > enhanced_threshold:
            best_match_index = similarities.index(max_cos_similarity)
            person_id = face_recognizer.known_person_ids[best_match_index]
            return True, person_id, combined_score

        return False, None, combined_score

    def is_similar_to_unknown(self, face_encoding: np.ndarray) -> Tuple[bool, str, float]:
        """Check if face is similar to any unknown faces we've seen"""
        from sklearn.metrics.pairwise import cosine_similarity

        for temp_id, known_encoding in self.unknown_faces.items():
            similarity = cosine_similarity([face_encoding], [known_encoding])[0][0]
            if similarity > FACE_SIMILARITY_THRESHOLD:
                return True, temp_id, similarity

        return False, "", 0.0

    def should_capture_face(self, face_encoding: np.ndarray) -> bool:
        """Enhanced duplicate prevention - check if we should capture this face"""
        # 1. Hash-based cooldown (fast check)
        face_hash = self.get_face_hash(face_encoding)
        if face_hash in self.capture_cooldowns:
            time_since_capture = time.time() - self.capture_cooldowns[face_hash]
            if time_since_capture < AUTO_CAPTURE_COOLDOWN:
                print(f"⏰ Face capture blocked - cooldown active ({time_since_capture:.1f}s < {AUTO_CAPTURE_COOLDOWN}s)")
                return False

        # 2. Check similarity to already known persons (prevent capturing known faces)
        is_known, person_id, confidence = self.is_similar_to_known(face_encoding)
        if is_known:
            print(f"🚫 Face capture blocked - person already known (ID: {person_id}, confidence: {confidence:.3f})")
            return False

        # 3. Check similarity to already captured unknown faces
        is_similar_unknown, temp_id, similarity = self.is_similar_to_unknown(face_encoding)
        if is_similar_unknown and similarity > FACE_SIMILARITY_THRESHOLD + 0.1:  # Stricter threshold
            print(f"🚫 Face capture blocked - similar unknown face already captured (ID: {temp_id}, similarity: {similarity:.3f})")
            return False

        # 4. Check if we already have too many photos in the auto_captured folder
        try:
            existing_photos = list(AUTO_CAPTURED_PATH.glob("*.jpg"))
            if len(existing_photos) > 100:  # Limit total photos to prevent disk space issues
                print(f"🚫 Face capture blocked - too many existing photos ({len(existing_photos)} > 100)")
                return False
        except:
            pass  # Continue if we can't check folder

        return True

    def capture_best_unknown_face(self, person: DetectedPerson) -> Optional[str]:
        """Capture the best quality face from candidates, preventing duplicates"""
        if not AUTO_CAPTURE_ENABLED or not person.capture_candidates:
            return None

        # Get the best quality candidate
        best_candidate = person.capture_candidates[0]  # Already sorted by quality

        if best_candidate.confidence < AUTO_CAPTURE_CONFIDENCE_THRESHOLD:
            return None

        if best_candidate.face_encoding is None:
            return None

        # Check if face should be captured (cooldown and duplicate prevention)
        if not self.should_capture_face(best_candidate.face_encoding):
            return None

        # Extract face region from best candidate
        x1, y1, x2, y2 = best_candidate.bbox
        face_height = y2 - y1
        if face_height < AUTO_CAPTURE_FACE_SIZE_MIN:
            return None

        frame = best_candidate.frame

        # Extract face with padding
        padding = 20
        face_x1 = max(0, x1 - padding)
        face_y1 = max(0, y1 - padding)
        face_x2 = min(frame.shape[1], x2 + padding)
        face_y2 = min(frame.shape[0], y2 + padding)

        face_region = frame[face_y1:face_y2, face_x1:face_x2]

        if face_region.size == 0:
            return None

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_hash = self.get_face_hash(best_candidate.face_encoding)
        filename = f"unknown_{timestamp}_{face_hash[:8]}_q{best_candidate.quality_score:.2f}.jpg"
        filepath = AUTO_CAPTURED_PATH / filename

        # Save image
        cv2.imwrite(str(filepath), face_region)

        # Update cooldown to prevent duplicates
        self.capture_cooldowns[face_hash] = time.time()

        # Check if similar to existing unknown faces
        is_similar, temp_id, similarity = self.is_similar_to_unknown(best_candidate.face_encoding)

        if not is_similar:
            # New unknown person
            self.temp_person_counter += 1
            temp_id = f"unknown_{self.temp_person_counter:04d}"
            self.unknown_faces[temp_id] = best_candidate.face_encoding

        print(f"📸 Auto-captured BEST quality face: {filename}")
        print(f"   📊 Quality Score: {best_candidate.quality_score:.3f}")
        print(f"   📊 Confidence: {best_candidate.confidence:.2f}")
        print(f"   📊 Total candidates evaluated: {len(person.capture_candidates)}")
        print(f"   🔖 Temp ID: {temp_id}")

        # Clear candidates after successful capture
        person.capture_candidates = []

        return temp_id

    def capture_unknown_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                           face_encoding: np.ndarray, confidence: float) -> Optional[str]:
        """Legacy method - kept for compatibility but redirects to quality system"""
        # This method is now primarily used for adding candidates
        # The actual capture happens via capture_best_unknown_face
        return None

    def cleanup_old_candidates(self, person: DetectedPerson, max_age_seconds: int = 30):
        """Clean up old capture candidates to prevent memory buildup"""
        if not person.capture_candidates:
            return

        current_time = time.time()
        # Remove candidates older than max_age_seconds
        person.capture_candidates = [
            candidate for candidate in person.capture_candidates
            if hasattr(candidate, 'timestamp') and (current_time - getattr(candidate, 'timestamp', 0)) < max_age_seconds
        ]

    def get_unknown_faces_count(self) -> int:
        """Get number of unique unknown faces detected"""
        return len(self.unknown_faces)

class MultiCameraSystem:
    """Advanced multi-camera system with improved detection"""

    def __init__(self):
        self.cameras: Dict[str, CameraStream] = {}
        self.detectors: Dict[str, YOLODetector] = {}
        self.detected_persons: Dict[str, DetectedPerson] = {}  # Global person tracking
        self.deduplication_system = FaceDeduplicationSystem()
        self.frame_queues: Dict[str, queue.Queue] = {}
        self.processing_threads: Dict[str, threading.Thread] = {}
        self.is_running = False

        # Camera mode for single camera
        self.camera_mode = CAMERA_MODE  # "entry", "exit", or "both"
        self.entry_records = {}  # Store entry records for exit matching
        self.person_session_tracker = {}  # Track persons to prevent duplicate counting
        self.photo_cooldowns = {}  # Track photo capture cooldowns

        # Statistics
        self.total_entries = 0
        self.total_exits = 0
        self.total_face_captures = 0
        self.repeat_visits = 0  # Track repeat visits
        self.person_visit_count = {}  # Track how many times each person has visited
        self.matched_exits = 0  # Track exits that matched with entries
        self.unmatched_exits = 0  # Track exits without matching entries

    def initialize_cameras(self) -> bool:
        """Initialize cameras based on configuration"""
        success = False

        if SINGLE_CAMERA_MODE:
            # Single camera mode
            cap = cv2.VideoCapture(SINGLE_CAMERA_INDEX)
            if cap.isOpened():
                camera = CameraStream(
                    index=SINGLE_CAMERA_INDEX,
                    name="Main Camera",
                    purpose="both",
                    cap=cap,
                    frame=None,
                    last_update=datetime.now(),
                    is_active=True
                )
                self.cameras["main"] = camera
                self.detectors["main"] = YOLODetector()
                self.frame_queues["main"] = queue.Queue(maxsize=5)
                success = True
                print(f"✅ Single camera initialized: {SINGLE_CAMERA_INDEX}")
        else:
            # Multi-camera mode
            for cam_key, cam_config in CAMERA_CONFIG.items():
                cap = cv2.VideoCapture(cam_config["index"])
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config["width"])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config["height"])
                    cap.set(cv2.CAP_PROP_FPS, cam_config["fps"])

                    camera = CameraStream(
                        index=cam_config["index"],
                        name=cam_config["name"],
                        purpose=cam_config["purpose"],
                        cap=cap,
                        frame=None,
                        last_update=datetime.now(),
                        is_active=True
                    )
                    self.cameras[cam_key] = camera
                    self.detectors[cam_key] = YOLODetector()
                    self.frame_queues[cam_key] = queue.Queue(maxsize=5)
                    success = True
                    print(f"✅ Camera initialized: {cam_config['name']} (index {cam_config['index']})")
                else:
                    print(f"❌ Failed to initialize camera: {cam_config['name']} (index {cam_config['index']})")

        return success

    def start(self) -> bool:
        """Start the multi-camera system"""
        if not self.initialize_cameras():
            print("❌ Failed to initialize any cameras")
            return False

        self.is_running = True

        # Start processing threads for each camera
        for cam_key in self.cameras.keys():
            thread = threading.Thread(target=self._process_camera, args=(cam_key,))
            thread.daemon = True
            thread.start()
            self.processing_threads[cam_key] = thread

        print("🚀 Multi-camera system started successfully")
        return True

    def stop(self):
        """Stop the multi-camera system"""
        self.is_running = False

        # Wait for threads to finish
        for thread in self.processing_threads.values():
            thread.join(timeout=2)

        # Release cameras
        for camera in self.cameras.values():
            if camera.cap:
                camera.cap.release()

        print("🛑 Multi-camera system stopped")

    def _process_camera(self, cam_key: str):
        """Process frames from a specific camera"""
        camera = self.cameras[cam_key]
        detector = self.detectors[cam_key]

        while self.is_running:
            try:
                # Capture frame
                ret, frame = camera.cap.read()
                if not ret:
                    continue

                camera.frame = frame.copy()
                camera.last_update = datetime.now()

                # Detect and track with current camera mode
                detector.set_camera_mode(self.camera_mode)  # Pass current mode to detector with history reset
                detections, events = detector.detect_and_track(frame)

                # Process detections for face recognition and auto-capture
                self._process_detections(cam_key, frame, detections)

                # Process events (entry/exit)
                self._process_events(cam_key, events)

                # Add to frame queue for display
                try:
                    self.frame_queues[cam_key].put_nowait(frame)
                except queue.Full:
                    pass  # Drop frame if queue is full

            except Exception as e:
                print(f"Error processing {cam_key}: {e}")

            time.sleep(0.033)  # ~30 FPS

    def _process_detections(self, cam_key: str, frame: np.ndarray, detections: List):
        """Process detections for face recognition and auto-capture"""
        if detections:
            print(f"👁️ Processing {len(detections)} detections from {cam_key}")

        for detection in detections:
            # Recognize face
            face_result = face_recognizer.recognize_face_in_bbox(frame, detection.bbox)
            recognized_name = face_result.get('person_name', 'Unknown')
            confidence = face_result.get('confidence', 0.0)

            if recognized_name != 'Unknown':
                print(f"✅ FACE RECOGNIZED: {recognized_name} (confidence: {confidence:.2f})")
            else:
                print(f"❌ Face not recognized (confidence: {confidence:.2f})")

            person_key = f"{cam_key}_{detection.track_id}"

            if person_key not in self.detected_persons:
                self.detected_persons[person_key] = DetectedPerson(
                    track_id=detection.track_id,
                    camera_source=cam_key,
                    person_id=face_result.get('person_id'),
                    person_name=face_result.get('person_name', 'Unknown'),
                    confidence=face_result.get('confidence', 0.0),
                    face_encoding=None,
                    last_seen=datetime.now(),
                    detection_count=1,
                    auto_captured=False
                )
            else:
                person = self.detected_persons[person_key]
                person.last_seen = datetime.now()
                person.detection_count += 1

                # Update person info if we get better recognition
                if face_result.get('confidence', 0) > person.confidence:
                    person.person_id = face_result.get('person_id')
                    person.person_name = face_result.get('person_name', 'Unknown')
                    person.confidence = face_result.get('confidence', 0.0)

            person = self.detected_persons[person_key]

            # Enhanced auto-capture system with quality assessment
            if (person.person_id is None and
                not person.auto_captured and
                person.detection_count >= MIN_FRAMES_FOR_FACE_CAPTURE and
                person.confidence > 0.5):

                # Extract face encoding for current frame
                face_region = face_recognizer.extract_face_from_bbox(frame, detection.bbox)
                if face_region is not None:
                    face_encoding = face_recognizer.extract_face_encoding(face_region)
                    if face_encoding is not None:
                        # Add current frame as capture candidate
                        self.deduplication_system.add_capture_candidate(
                            person, frame, detection.bbox, person.confidence, face_encoding
                        )

                        # If we have enough candidates (5 frames) or person has been tracked long enough,
                        # capture the best quality photo
                        if (len(person.capture_candidates) >= 5 or
                            person.detection_count >= MIN_FRAMES_FOR_FACE_CAPTURE + 5):

                            temp_id = self.deduplication_system.capture_best_unknown_face(person)
                            if temp_id:
                                person.auto_captured = True
                                person.face_encoding = face_encoding
                                self.total_face_captures += 1
                                print(f"✅ Successfully captured best quality photo for person {temp_id}")

            # Continue collecting candidates for unknown faces even after initial frames
            elif (person.person_id is None and
                  not person.auto_captured and
                  person.detection_count < MIN_FRAMES_FOR_FACE_CAPTURE + 10 and
                  person.confidence > 0.3):  # Lower threshold for candidate collection

                face_region = face_recognizer.extract_face_from_bbox(frame, detection.bbox)
                if face_region is not None:
                    face_encoding = face_recognizer.extract_face_encoding(face_region)
                    if face_encoding is not None:
                        # Keep collecting candidates
                        self.deduplication_system.add_capture_candidate(
                            person, frame, detection.bbox, person.confidence, face_encoding
                        )

    def _process_events(self, cam_key: str, events: List[Dict]):
        """Process entry/exit events based on camera mode"""
        camera = self.cameras[cam_key]

        for event in events:
            person_key = f"{cam_key}_{event['track_id']}"
            person = self.detected_persons.get(person_key)

            if SINGLE_CAMERA_MODE:
                # Single camera mode - respect current mode
                if self.camera_mode == 'entry' and event['type'] == 'entry':
                    self._process_entry_event(cam_key, event, person)
                elif self.camera_mode == 'exit' and event['type'] == 'exit':
                    self._process_exit_event(cam_key, event, person)
                elif self.camera_mode == 'both':
                    # Both mode - handle both entry and exit
                    if event['type'] == 'entry':
                        self._process_entry_event(cam_key, event, person)
                    elif event['type'] == 'exit':
                        self._process_exit_event(cam_key, event, person)
            else:
                # Multi-camera mode - respect camera purpose
                if event['type'] == 'entry' and camera.purpose in ['entry', 'both']:
                    self._process_entry_event(cam_key, event, person)
                elif event['type'] == 'exit' and camera.purpose in ['exit', 'both']:
                    self._process_exit_event(cam_key, event, person)

    def _process_entry_event(self, cam_key: str, event: Dict, person: Optional[DetectedPerson]):
        """Process entry event with deduplication"""
        print(f"🚪 ENTRY EVENT triggered for {person.person_name if person else 'Unknown'}")

        # Get person identification
        person_name = person.person_name if person else 'Unknown'
        session_key = self._get_session_key(person)
        current_time = datetime.now()

        # Check if this is a duplicate entry (same person within cooldown period)
        is_duplicate = False
        if session_key in self.person_session_tracker:
            last_entry_time = self.person_session_tracker[session_key].get('last_entry')
            if last_entry_time and (current_time - last_entry_time).total_seconds() < 30:  # 30 second cooldown
                is_duplicate = True
                print(f"⏰ Duplicate entry detected for {person_name} (within 30 seconds)")

        # Track visits for all persons (including unknown)
        # For unknown persons, use track_id as identifier
        visit_key = person_name if person_name != 'Unknown' else f"Unknown_Track_{event.get('track_id', 0)}"

        if visit_key not in self.person_visit_count:
            # First visit ever
            self.person_visit_count[visit_key] = 1
            print(f"🆕 FIRST VISIT: {visit_key}")
            # Note: total_entries is now managed by the YOLO detector, not here
            # self.total_entries += 1  # Removed to prevent double counting
        elif not is_duplicate:
            # Repeat visit (not a duplicate)
            self.person_visit_count[visit_key] += 1
            self.repeat_visits += 1
            # Note: total_entries is now managed by the YOLO detector, not here
            # self.total_entries += 1  # Removed to prevent double counting
            print(f"🔄 REPEAT VISIT #{self.person_visit_count[visit_key]}: {visit_key}")
            print(f"📊 Total repeat visits counter: {self.repeat_visits}")
        else:
            # Duplicate within cooldown - don't count
            print(f"⏭️ Skipping duplicate count for {visit_key} (cooldown active)")

        # Update session tracker
        if session_key not in self.person_session_tracker:
            self.person_session_tracker[session_key] = {}
        self.person_session_tracker[session_key]['last_entry'] = current_time
        self.person_session_tracker[session_key]['person_name'] = person_name

        # Store entry record for later exit matching (enhanced)
        if person:
            # Create comprehensive entry record
            entry_key = person.person_id if person.person_id else f"track_{event.get('track_id', 0)}"
            self.entry_records[entry_key] = {
                'timestamp': current_time,
                'person_name': person.person_name,
                'person_id': person.person_id,
                'confidence': person.confidence,
                'face_encoding': person.face_encoding,
                'track_id': event.get('track_id', 0),
                'entry_location': event.get('position', (0, 0))
            }
            print(f"📝 Entry record saved: {person.person_name} (ID: {entry_key})")

        # Capture faces in entry or both mode for better quality
        # Always capture entry photos regardless of whether person is known or unknown
        if self.camera_mode in ['entry', 'both']:
            self._capture_entry_photo(cam_key, event, person)

        self._log_event(cam_key, event, 'entry')
        camera = self.cameras[cam_key]
        print(f"🟢 ENTRY detected on {camera.name} - {person.person_name if person else 'Unknown'} - Total: {self.total_entries}")

    def _process_exit_event(self, cam_key: str, event: Dict, person: Optional[DetectedPerson]):
        """Process exit event with improved identification and matching"""
        print(f"🚪 EXIT EVENT triggered for {person.person_name if person else 'Unknown'}")

        # Get person identification
        person_name = person.person_name if person else 'Unknown'
        session_key = self._get_session_key(person)
        current_time = datetime.now()

        # Check if this is a duplicate exit (same person within cooldown period)
        is_duplicate = False
        if session_key in self.person_session_tracker:
            last_exit_time = self.person_session_tracker[session_key].get('last_exit')
            if last_exit_time and (current_time - last_exit_time).total_seconds() < 30:  # 30 second cooldown
                is_duplicate = True
                print(f"⏰ Duplicate exit detected for {person_name} (within 30 seconds)")

        if not is_duplicate:
            # Note: total_exits is now managed by the YOLO detector, not here
            # self.total_exits += 1  # Removed to prevent double counting

            # Try to match exit with entry records
            matched_entry = self._find_matching_entry(person, event)
            if matched_entry:
                self.matched_exits += 1
                duration = (current_time - matched_entry['timestamp']).total_seconds()
                print(f"🎯 EXIT MATCHED with entry: {matched_entry['person_name']} (entered at {matched_entry['timestamp'].strftime('%H:%M:%S')})")
                print(f"⏱️ Duration inside: {duration:.1f} seconds")
                print(f"📊 Matched exits: {self.matched_exits}")

                # Remove matched entry record to prevent duplicate matching
                entry_keys_to_remove = []
                for entry_key, entry_record in self.entry_records.items():
                    if entry_record == matched_entry:
                        entry_keys_to_remove.append(entry_key)

                for key in entry_keys_to_remove:
                    del self.entry_records[key]
                    print(f"🗑️ Removed matched entry record: {key}")
            else:
                self.unmatched_exits += 1
                print(f"❓ EXIT without matching entry: {person_name}")
                print(f"📊 Unmatched exits: {self.unmatched_exits}")

            # Capture exit photo for identification
            self._capture_exit_photo(cam_key, event, person)

            print(f"🔴 EXIT processed: {person_name} - Total exits: {self.total_exits}")
        else:
            print(f"⏭️ Skipping duplicate exit for {person_name}")

        # Update session tracker
        if session_key not in self.person_session_tracker:
            self.person_session_tracker[session_key] = {}
        self.person_session_tracker[session_key]['last_exit'] = current_time
        self.person_session_tracker[session_key]['person_name'] = person_name

        self._log_event(cam_key, event, 'exit')
        camera = self.cameras[cam_key]

    def _find_matching_entry(self, person: Optional[DetectedPerson], event: Dict) -> Optional[Dict]:
        """Find matching entry record for this exit with improved logic"""
        if not person:
            return None

        print(f"🔍 Looking for entry match for {person.person_name} (track: {event.get('track_id', 'Unknown')})")
        print(f"📋 Available entry records: {len(self.entry_records)}")

        # Method 1: Direct person_id match (most reliable for known persons)
        if person.person_id and person.person_id in self.entry_records:
            match = self.entry_records[person.person_id]
            print(f"✅ Direct person_id match found: {match['person_name']}")
            return match

        # Method 2: Track ID match (for same tracking session)
        track_key = f"track_{event.get('track_id', 0)}"
        if track_key in self.entry_records:
            match = self.entry_records[track_key]
            print(f"✅ Track ID match found: {match['person_name']} (track: {match['track_id']})")
            return match

        # Method 3: Match by person name (for recognized faces)
        if person.person_name and person.person_name != 'Unknown':
            for entry_key, entry_record in self.entry_records.items():
                if entry_record['person_name'] == person.person_name:
                    print(f"✅ Name match found: {person.person_name}")
                    return entry_record

        # Method 4: Face encoding similarity (fallback)
        if person.face_encoding is not None:
            best_match = None
            best_similarity = 0.0
            best_key = None

            for entry_key, entry_record in self.entry_records.items():
                if entry_record.get('face_encoding') is not None:
                    from sklearn.metrics.pairwise import cosine_similarity
                    try:
                        similarity = cosine_similarity([person.face_encoding], [entry_record['face_encoding']])[0][0]

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = entry_record
                            best_key = entry_key

                    except Exception as e:
                        print(f"❌ Error computing similarity: {e}")

            if best_match and best_similarity > 0.7:  # Lower threshold for exit matching
                print(f"✅ Face similarity match found: {best_match['person_name']} (similarity: {best_similarity:.2f})")
                return best_match

        print(f"❌ No matching entry found for exit")
        return None

    def _capture_exit_photo(self, cam_key: str, event: Dict, person: Optional[DetectedPerson]):
        """Capture exit photo for identification and verification"""
        if not person:
            print("❌ No person object for exit photo capture")
            return

        camera = self.cameras[cam_key]
        frame = camera.frame

        if frame is None:
            print("❌ No frame available for exit photo capture")
            return

        print(f"📸 Capturing exit photo for {person.person_name}")

        # Extract face region for exit photo
        bbox = event.get('bbox', {})
        print(f"📦 Exit BBox data: {bbox}")

        if all(k in bbox for k in ['x1', 'y1', 'x2', 'y2']):
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])

            # Add padding
            padding = 30
            face_x1 = max(0, x1 - padding)
            face_y1 = max(0, y1 - padding)
            face_x2 = min(frame.shape[1], x2 + padding)
            face_y2 = min(frame.shape[0], y2 + padding)

            face_region = frame[face_y1:face_y2, face_x1:face_x2]

            if face_region.size > 0:
                # Generate exit photo filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                person_name = person.person_name.replace(' ', '_') if person.person_name != 'Unknown' else 'unknown'
                filename = f"exit_{person_name}_{timestamp}.jpg"
                filepath = AUTO_CAPTURED_PATH / filename

                # Save exit photo
                success = cv2.imwrite(str(filepath), face_region, [cv2.IMWRITE_JPEG_QUALITY, 95])

                if success:
                    print(f"✅ Exit photo saved: {filepath}")
                else:
                    print(f"❌ Failed to save exit photo to: {filepath}")
            else:
                print("❌ Exit face region is empty!")
        else:
            print(f"❌ Exit BBox missing keys! Got: {bbox.keys() if bbox else 'None'}")

    def _get_session_key(self, person: Optional[DetectedPerson]) -> str:
        """Generate a session key for person tracking"""
        if person and person.person_id:
            return f"person_{person.person_id}"
        elif person and person.person_name and person.person_name != 'Unknown':
            return f"name_{person.person_name}"
        else:
            # For unknown persons, use face encoding similarity or track_id
            return f"track_{person.track_id}_{person.camera_source}" if person else "unknown"

    def _capture_entry_photo(self, cam_key: str, event: Dict, person: Optional[DetectedPerson]):
        """Capture entry photo - SIMPLIFIED VERSION"""
        if not person:
            print("❌ No person object for photo capture")
            return

        camera = self.cameras[cam_key]
        frame = camera.frame

        if frame is None:
            print("❌ No frame available for photo capture")
            return

        if person.auto_captured:
            print(f"⏭️ Photo already captured for {person.person_name}")
            return

        print(f"📸 Attempting to capture entry photo for {person.person_name}")

        # Extract face region for better quality photo
        bbox = event.get('bbox', {})
        print(f"📦 BBox data: {bbox}")

        if all(k in bbox for k in ['x1', 'y1', 'x2', 'y2']):
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            print(f"📏 BBox coords: ({x1},{y1}) to ({x2},{y2})")

            # Add simple padding
            padding = 30
            face_x1 = max(0, x1 - padding)
            face_y1 = max(0, y1 - padding)
            face_x2 = min(frame.shape[1], x2 + padding)
            face_y2 = min(frame.shape[0], y2 + padding)

            face_region = frame[face_y1:face_y2, face_x1:face_x2]
            print(f"🖼️ Face region shape: {face_region.shape if face_region.size > 0 else 'EMPTY'}")

            if face_region.size > 0:
                # Generate simple filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                person_name = person.person_name.replace(' ', '_') if person.person_name != 'Unknown' else 'unknown'
                filename = f"entry_{person_name}_{timestamp}.jpg"
                filepath = AUTO_CAPTURED_PATH / filename

                # Save entry photo
                success = cv2.imwrite(str(filepath), face_region, [cv2.IMWRITE_JPEG_QUALITY, 95])

                if success:
                    person.auto_captured = True
                    self.total_face_captures += 1
                    print(f"✅ Entry photo saved: {filepath}")
                else:
                    print(f"❌ Failed to save photo to: {filepath}")
            else:
                print("❌ Face region is empty!")
        else:
            print(f"❌ BBox missing keys! Got: {bbox.keys() if bbox else 'None'}")

    def _assess_face_quality(self, face_region: np.ndarray) -> bool:
        """Assess face quality using multiple metrics"""
        try:
            # Convert to grayscale for quality assessment
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

            # 1. Blur detection using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            # 2. Brightness assessment
            mean_brightness = np.mean(gray)

            # 3. Size check
            height, width = face_region.shape[:2]

            # Quality thresholds (from config)
            blur_threshold = AUTO_CAPTURE_BLUR_THRESHOLD
            brightness_min = AUTO_CAPTURE_BRIGHTNESS_MIN
            brightness_max = AUTO_CAPTURE_BRIGHTNESS_MAX
            min_size = AUTO_CAPTURE_FACE_SIZE_MIN

            # Check all quality criteria
            is_sharp = blur_score > blur_threshold
            is_well_lit = brightness_min <= mean_brightness <= brightness_max
            is_large_enough = min(height, width) >= min_size

            quality_passed = is_sharp and is_well_lit and is_large_enough

            if not quality_passed:
                print(f"⚠️ Face quality check failed: blur={blur_score:.1f}, brightness={mean_brightness:.1f}, size={min(height, width)}")

            return quality_passed

        except Exception as e:
            print(f"Error in face quality assessment: {e}")
            return True  # Default to True if assessment fails

    def set_camera_mode(self, mode: str) -> bool:
        """Set camera mode for single camera operation"""
        if mode in ['entry', 'exit', 'both']:
            self.camera_mode = mode
            print(f"📹 Camera mode changed to: {mode}")

            # Update all detectors with new mode
            for detector in self.detectors.values():
                detector.set_camera_mode(mode)

            # Clear session tracker when mode changes to prevent issues
            self.person_session_tracker.clear()
            print("🔄 Session tracker cleared for mode change")

            return True
        return False

    def _log_event(self, cam_key: str, event: Dict, event_type: str):
        """Log event to database"""
        try:
            person_key = f"{cam_key}_{event['track_id']}"
            person = self.detected_persons.get(person_key)

            bbox_dict = event.get('bbox', {"x1": 0, "y1": 0, "x2": 0, "y2": 0})

            db.add_detection_log(
                person_id=person.person_id if person else None,
                detection_type=event_type,
                confidence=event.get('confidence', 0.0),
                detection_method=f"multi_camera_{cam_key}",
                bounding_box=bbox_dict,
                person_name=person.person_name if person else "Unknown"
            )
        except Exception as e:
            print(f"Error logging event: {e}")

    def get_latest_frame(self, cam_key: str = None) -> Optional[np.ndarray]:
        """Get latest frame from specified camera or main camera"""
        if cam_key is None:
            cam_key = list(self.cameras.keys())[0]  # Get first camera

        if cam_key in self.cameras:
            return self.cameras[cam_key].frame
        return None

    def get_processed_frame(self, cam_key: str = None) -> Optional[np.ndarray]:
        """Get processed frame with detections drawn"""
        frame = self.get_latest_frame(cam_key)
        if frame is None:
            return None

        if cam_key is None:
            cam_key = list(self.cameras.keys())[0]

        detector = self.detectors[cam_key]
        detections, _ = detector.detect_and_track(frame)
        processed_frame = detector.draw_detections(frame, detections)

        # Add system info
        self._draw_system_info(processed_frame, cam_key)

        return processed_frame

    def _draw_system_info(self, frame: np.ndarray, cam_key: str):
        """Draw system information on frame based on current mode"""
        camera = self.cameras[cam_key]

        # Background for info
        cv2.rectangle(frame, (10, frame.shape[0] - 100), (300, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, frame.shape[0] - 100), (300, frame.shape[0] - 10), (255, 255, 255), 2)

        # Always show camera name
        y_pos = frame.shape[0] - 80
        cv2.putText(frame, f"Camera: {camera.name}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_pos += 20

        # Show different info based on mode
        # For Entry/Exit modes, use live detector occupancy; for Both mode use calculation
        if self.camera_mode in ['entry', 'exit'] and self.detectors:
            # Get current occupancy from detector (live count of people in camera)
            detector = list(self.detectors.values())[0]  # Get first (and only) detector
            current_occupancy = detector.current_occupancy
        else:
            # Both mode uses entry - exit calculation
            current_occupancy = max(0, self.total_entries - self.total_exits)

        # Get actual totals from detector for display
        if self.camera_mode in ['entry', 'exit'] and self.detectors:
            detector = list(self.detectors.values())[0]
            display_entries = detector.total_entries
            display_exits = detector.total_exits
        else:
            # Both mode uses multi-camera system totals
            display_entries = self.total_entries
            display_exits = self.total_exits

        if self.camera_mode == 'entry':
            # Entry mode: only show entry count and current inside
            cv2.putText(frame, f"Entries: {display_entries}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 20
            cv2.putText(frame, f"Current Inside: {current_occupancy}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        elif self.camera_mode == 'exit':
            # Exit mode: only show exit count and current inside
            cv2.putText(frame, f"Exits: {display_exits}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_pos += 20
            cv2.putText(frame, f"Current Inside: {current_occupancy}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        elif self.camera_mode == 'both':
            # Both mode: show entries, exits, and current inside
            cv2.putText(frame, f"Entries: {display_entries} | Exits: {display_exits}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 20
            cv2.putText(frame, f"Current Inside: {current_occupancy}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def get_statistics(self) -> Dict:
        """Get system statistics"""
        active_persons = len([p for p in self.detected_persons.values()
                            if (datetime.now() - p.last_seen).seconds < 30])

        # Calculate current occupancy based on mode
        if self.camera_mode in ['entry', 'exit'] and self.detectors:
            # Get current occupancy from detector (live count of people in camera)
            detector = list(self.detectors.values())[0]  # Get first (and only) detector
            current_occupancy = detector.current_occupancy
        else:
            # Both mode uses entry - exit calculation
            current_occupancy = max(0, self.total_entries - self.total_exits)

        # Get actual totals from detector for Entry/Exit modes
        if self.camera_mode in ['entry', 'exit'] and self.detectors:
            detector = list(self.detectors.values())[0]
            total_entries = detector.total_entries
            total_exits = detector.total_exits
        else:
            # Both mode uses multi-camera system totals
            total_entries = self.total_entries
            total_exits = self.total_exits

        return {
            "total_entries": total_entries,
            "total_exits": total_exits,
            "current_occupancy": current_occupancy,
            "active_cameras": len([c for c in self.cameras.values() if c.is_active]),
            "active_persons": active_persons,
            "unknown_faces_detected": self.deduplication_system.get_unknown_faces_count(),
            "auto_captured_faces": self.total_face_captures,
            "repeat_visits": self.repeat_visits,
            "matched_exits": self.matched_exits,
            "unmatched_exits": self.unmatched_exits,
            "pending_entries": len(self.entry_records)
        }

    def reset_counters(self):
        """Reset all counters"""
        self.total_entries = 0
        self.total_exits = 0
        self.total_face_captures = 0
        self.repeat_visits = 0
        self.matched_exits = 0
        self.unmatched_exits = 0
        self.person_visit_count.clear()
        self.entry_records.clear()
        self.person_session_tracker.clear()
        self.photo_cooldowns.clear()
        self.detected_persons.clear()
        print("🔄 All counters and records reset")

# Global multi-camera system instance
multi_camera_system = MultiCameraSystem()