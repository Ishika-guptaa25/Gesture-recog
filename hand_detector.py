"""
Hand detection module using MediaPipe
Detects hand landmarks and provides hand tracking data
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class Hand:
    """Represents a single hand with landmarks and metadata"""

    def __init__(self, landmarks, handedness, confidence):
        self.landmarks = landmarks  # 21 landmarks
        self.handedness = handedness  # 'Left' or 'Right'
        self.confidence = confidence
        self.palm_center = self._calculate_palm_center()
        self.fingers = self._detect_fingers()

    def _calculate_palm_center(self) -> np.ndarray:
        """Calculate center of palm from landmarks"""
        palm_landmarks = self.landmarks[:5]
        return np.mean(palm_landmarks, axis=0)

    def _detect_fingers(self) -> Dict[str, np.ndarray]:
        """Detect finger positions"""
        return {
            'thumb': self.landmarks[4],
            'index': self.landmarks[8],
            'middle': self.landmarks[12],
            'ring': self.landmarks[16],
            'pinky': self.landmarks[20],
        }

    def get_hand_size(self) -> float:
        """Calculate hand bounding box size"""
        x_coords = self.landmarks[:, 0]
        y_coords = self.landmarks[:, 1]
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)
        return np.sqrt(width * height)


class HandDetector:
    """Hand detection using MediaPipe Hands"""

    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2):
        """
        Initialize hand detector

        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            max_num_hands: Maximum number of hands to detect
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.image_height = 0
        self.image_width = 0

        logger.info(f"HandDetector initialized with max_num_hands={max_num_hands}")

    def detect(self, frame: np.ndarray) -> List[Hand]:
        """
        Detect hands in frame

        Args:
            frame: Input image frame (BGR format from OpenCV)

        Returns:
            List of detected Hand objects
        """
        self.image_height, self.image_width = frame.shape[:2]

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hands = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Convert landmarks to numpy array with normalized coordinates
                landmarks_array = np.array([
                    [lm.x, lm.y, lm.z] for lm in landmarks.landmark
                ])

                # Convert to pixel coordinates
                landmarks_array[:, 0] *= self.image_width
                landmarks_array[:, 1] *= self.image_height

                hand = Hand(
                    landmarks=landmarks_array[:, :2],  # Use only x, y
                    handedness=handedness.classification[0].label,
                    confidence=handedness.classification[0].score
                )
                hands.append(hand)

        return hands

    def draw_landmarks(self, frame: np.ndarray, hands: List[Hand],
                       circle_color=(0, 255, 0), line_color=(255, 0, 0)) -> np.ndarray:
        """
        Draw hand landmarks on frame

        Args:
            frame: Input frame
            hands: List of detected hands
            circle_color: Color for landmark circles (BGR)
            line_color: Color for connection lines (BGR)

        Returns:
            Frame with drawn landmarks
        """
        frame_copy = frame.copy()

        for hand in hands:
            # Draw circles for landmarks
            for landmark in hand.landmarks:
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(frame_copy, (x, y), 4, circle_color, -1)

            # Draw connections between landmarks
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (0, 17),  # Palm
            ]

            for start, end in connections:
                x1, y1 = int(hand.landmarks[start][0]), int(hand.landmarks[start][1])
                x2, y2 = int(hand.landmarks[end][0]), int(hand.landmarks[end][1])
                cv2.line(frame_copy, (x1, y1), (x2, y2), line_color, 2)

        return frame_copy

    def get_hand_positions(self, hands: List[Hand]) -> List[Tuple[float, float]]:
        """Get palm center positions of all hands"""
        return [tuple(hand.palm_center.astype(int)) for hand in hands]

    def release(self):
        """Release MediaPipe resources"""
        self.hands.close()
        logger.info("HandDetector resources released")