"""
Gesture recognition module
Recognizes hand gestures from detected hand landmarks
"""

import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class GestureRecognizer:
    """Recognizes hand gestures from landmarks"""

    def __init__(self, threshold=0.7):
        """
        Initialize gesture recognizer

        Args:
            threshold: Confidence threshold for gesture recognition (0-1)
        """
        self.threshold = threshold
        self.last_gesture = None
        self.gesture_cooldown = 0

    def recognize(self, hand) -> Optional[str]:
        """
        Recognize gesture from hand landmarks

        Args:
            hand: Hand object with landmarks

        Returns:
            Gesture name or None
        """
        if hand is None:
            return None

        # Update cooldown
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return self.last_gesture

        landmarks = hand.landmarks
        fingers = hand.fingers

        # Get hand metrics
        hand_size = hand.get_hand_size()

        # Check gestures (order matters - more specific first)
        gesture_checks = [
            ('peace_sign', self._is_peace_sign),
            ('thumbs_up', self._is_thumbs_up),
            ('pointing', self._is_pointing),
            ('open_palm', self._is_open_palm),
            ('fist', self._is_fist),
            ('heart', self._is_heart),
        ]

        for gesture_name, check_func in gesture_checks:
            confidence = check_func(landmarks, fingers)
            if confidence > self.threshold:
                self.last_gesture = gesture_name
                self.gesture_cooldown = 5  # Cooldown frames
                logger.debug(f"Gesture detected: {gesture_name} (confidence: {confidence:.2f})")
                return gesture_name

        return None

    def _is_peace_sign(self, landmarks: np.ndarray, fingers: Dict) -> float:
        """Detect peace sign (index and middle fingers up, others down)"""
        # Check if index and middle are extended
        index_extended = landmarks[8][1] < landmarks[6][1]
        middle_extended = landmarks[12][1] < landmarks[10][1]

        # Check if ring and pinky are down
        ring_down = landmarks[16][1] > landmarks[14][1]
        pinky_down = landmarks[20][1] > landmarks[18][1]

        if index_extended and middle_extended and ring_down and pinky_down:
            # Calculate finger distance
            distance = np.linalg.norm(
                landmarks[8][:2] - landmarks[12][:2]
            )
            return min(1.0, distance / 50)

        return 0.0

    def _is_thumbs_up(self, landmarks: np.ndarray, fingers: Dict) -> float:
        """Detect thumbs up gesture"""
        # Thumb should be up (low y value relative to wrist)
        thumb_up = landmarks[4][1] < landmarks[2][1] - 30

        # Other fingers should be curled (high y values)
        fingers_curled = (
                landmarks[8][1] > landmarks[6][1] and
                landmarks[12][1] > landmarks[10][1] and
                landmarks[16][1] > landmarks[14][1] and
                landmarks[20][1] > landmarks[18][1]
        )

        if thumb_up and fingers_curled:
            # Check hand orientation (palm facing down)
            palm_down = landmarks[9][1] > landmarks[0][1]
            return 0.9 if palm_down else 0.6

        return 0.0

    def _is_pointing(self, landmarks: np.ndarray, fingers: Dict) -> float:
        """Detect pointing gesture (index finger extended)"""
        # Index finger should be extended
        index_extended = landmarks[8][1] < landmarks[6][1] - 20

        # Other fingers should be relatively closed
        other_down = (
                landmarks[12][1] > landmarks[10][1] and
                landmarks[16][1] > landmarks[14][1] and
                landmarks[20][1] > landmarks[18][1]
        )

        if index_extended and other_down:
            # Calculate pointing direction strength
            pointing_strength = (landmarks[6][1] - landmarks[8][1]) / 100
            return min(1.0, pointing_strength)

        return 0.0

    def _is_open_palm(self, landmarks: np.ndarray, fingers: Dict) -> float:
        """Detect open palm gesture"""
        # All fingers should be extended
        fingers_extended = (
                landmarks[4][1] < landmarks[2][1] and  # Thumb
                landmarks[8][1] < landmarks[6][1] and  # Index
                landmarks[12][1] < landmarks[10][1] and  # Middle
                landmarks[16][1] < landmarks[14][1] and  # Ring
                landmarks[20][1] < landmarks[18][1]  # Pinky
        )

        if fingers_extended:
            # Check finger spread
            finger_positions = [
                landmarks[4][:2], landmarks[8][:2], landmarks[12][:2],
                landmarks[16][:2], landmarks[20][:2]
            ]

            distances = []
            for i in range(len(finger_positions)):
                for j in range(i + 1, len(finger_positions)):
                    dist = np.linalg.norm(finger_positions[i] - finger_positions[j])
                    distances.append(dist)

            avg_distance = np.mean(distances)
            spread_score = min(1.0, avg_distance / 100)

            return spread_score if spread_score > 0.5 else 0.0

        return 0.0

    def _is_fist(self, landmarks: np.ndarray, fingers: Dict) -> float:
        """Detect fist gesture (all fingers curled)"""
        # All fingers should be curled (low position relative to knuckles)
        all_curled = (
                landmarks[4][1] > landmarks[2][1] and
                landmarks[8][1] > landmarks[6][1] and
                landmarks[12][1] > landmarks[10][1] and
                landmarks[16][1] > landmarks[14][1] and
                landmarks[20][1] > landmarks[18][1]
        )

        return 0.85 if all_curled else 0.0

    def _is_heart(self, landmarks: np.ndarray, fingers: Dict) -> float:
        """Detect heart shape (both index and middle make thumbs)"""
        # Both hands would form hearts, so this is simplified
        # Check if thumbs and index fingers are extended
        thumbs_extended = landmarks[4][1] < landmarks[2][1]
        index_extended = landmarks[8][1] < landmarks[6][1]

        if thumbs_extended and index_extended:
            # Check proximity
            thumb_index_distance = np.linalg.norm(
                landmarks[4][:2] - landmarks[8][:2]
            )

            if thumb_index_distance < 50:
                return 0.8

        return 0.0

    def get_all_gestures(self) -> Dict[str, str]:
        """Get description of all supported gestures"""
        return {
            'peace_sign': '✌️ Peace/Victory sign',
            'thumbs_up': '👍 Thumbs up',
            'pointing': '👉 Pointing finger',
            'open_palm': '✋ Open palm',
            'fist': '✊ Closed fist',
            'heart': '❤️ Heart shape',
        }