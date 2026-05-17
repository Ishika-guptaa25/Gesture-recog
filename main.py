"""
AR Hand Effects Generator - Complete Application
Real-time hand detection with visual effects
Single file - Easy to understand and modify
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import logging
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION - Edit these!
# ============================================================================

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
TARGET_FPS = 30
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
EFFECT_INTENSITY = 0.8
PARTICLE_COUNT = 100
TRAIL_LENGTH = 30
SHOW_HAND_LANDMARKS = True
SHOW_GESTURES = True
SHOW_FPS = True

os.makedirs("screenshots", exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CLASS 1: HAND DETECTION
# ============================================================================

class HandDetector:
    """Detect hands using MediaPipe"""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.image_height = 0
        self.image_width = 0

    def detect(self, frame):
        """Detect hands in frame"""
        self.image_height, self.image_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hands = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmarks_array = np.array([[lm.x * self.image_width, lm.y * self.image_height]
                                           for lm in landmarks.landmark])
                hands.append({
                    'landmarks': landmarks_array,
                    'handedness': handedness.classification[0].label,
                    'confidence': handedness.classification[0].score,
                    'palm_center': np.mean(landmarks_array[:5], axis=0)
                })
        return hands

    def draw_landmarks(self, frame, hands):
        """Draw hand landmarks on frame"""
        for hand in hands:
            landmarks = hand['landmarks'].astype(int)

            # Draw circles
            for point in landmarks:
                cv2.circle(frame, tuple(point), 4, (0, 255, 0), -1)

            # Draw connections
            connections = [
                (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
                (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                (13,17),(17,18),(18,19),(19,20),(0,17)
            ]

            for start, end in connections:
                p1 = tuple(landmarks[start])
                p2 = tuple(landmarks[end])
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

        return frame

    def release(self):
        self.hands.close()

# ============================================================================
# CLASS 2: GESTURE RECOGNITION
# ============================================================================

class GestureRecognizer:
    """Recognize hand gestures"""

    def recognize(self, hand):
        """Recognize gesture from landmarks"""
        landmarks = hand['landmarks']

        # Peace sign
        if landmarks[8][1] < landmarks[6][1] and landmarks[12][1] < landmarks[10][1]:
            if landmarks[16][1] > landmarks[14][1] and landmarks[20][1] > landmarks[18][1]:
                return "peace_sign"

        # Thumbs up
        if landmarks[4][1] < landmarks[2][1] - 30:
            if all(landmarks[i][1] > landmarks[i-2][1] for i in [8,12,16,20]):
                return "thumbs_up"

        # Pointing
        if landmarks[8][1] < landmarks[6][1] - 20:
            if all(landmarks[i][1] > landmarks[i-2][1] for i in [12,16,20]):
                return "pointing"

        # Open palm
        if all(landmarks[i][1] < landmarks[i-2][1] for i in [4,8,12,16,20]):
            return "open_palm"

        # Fist
        if all(landmarks[i][1] > landmarks[i-2][1] for i in [4,8,12,16,20]):
            return "fist"

        return None

# ============================================================================
# CLASS 3: VISUAL EFFECTS
# ============================================================================

class EffectsEngine:
    """Render visual effects"""

    def __init__(self):
        self.particles = []
        self.trails = {}
        self.current_effect = 'particles'
        self.frame_count = 0
        self.colors = {
            'neon': [(0,255,255), (255,0,255), (255,255,0), (0,255,0)],
            'fire': [(0,0,255), (0,165,255), (0,255,255), (0,255,0)],
            'ice': [(255,128,0), (0,255,255), (255,255,255), (0,200,255)],
            'pastel': [(255,179,198), (255,223,186), (255,255,200), (186,255,201)]
        }
        self.palette = 'neon'
        self.color_index = 0

    def render_particles(self, frame, hand_pos):
        """Render particle burst effect"""
        if self.frame_count % 3 == 0:
            for _ in range(5):
                angle = np.random.uniform(0, 2*np.pi)
                speed = np.random.uniform(2, 6)
                vx, vy = speed*np.cos(angle), speed*np.sin(angle)
                color = self.colors[self.palette][len(self.particles) % 4]
                self.particles.append({
                    'x': hand_pos[0], 'y': hand_pos[1],
                    'vx': vx, 'vy': vy, 'color': color,
                    'lifetime': 30, 'init_life': 30
                })

        for p in self.particles[:]:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['lifetime'] -= 1
            p['vx'] *= 0.95
            p['vy'] *= 0.95

            if p['lifetime'] <= 0:
                self.particles.remove(p)
            else:
                alpha = p['lifetime'] / p['init_life']
                size = int(4 * alpha)
                x, y = int(p['x']), int(p['y'])
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    cv2.circle(frame, (x, y), max(1, size), p['color'], -1)

        return frame

    def render_trail(self, frame, hand_id, hand_pos):
        """Render motion trail effect"""
        if hand_id not in self.trails:
            self.trails[hand_id] = deque(maxlen=TRAIL_LENGTH)

        hand_pos_tuple = tuple(hand_pos.astype(int)) if isinstance(hand_pos, np.ndarray) else hand_pos
        self.trails[hand_id].append(hand_pos_tuple)
        trail = list(self.trails[hand_id])

        for i in range(len(trail)-1):
            alpha = i / len(trail)
            color = self.colors[self.palette][i % 4]
            thickness = int(5 * alpha)
            if thickness > 0:
                cv2.line(frame, trail[i], trail[i+1], color, thickness)

        return frame

    def render_glow(self, frame, hand_pos):
        """Render glow effect around hand"""
        overlay = frame.copy()
        color = self.colors[self.palette][self.color_index]

        for r in range(50, 0, -5):
            alpha = (50-r) / 50 * 0.3
            cv2.circle(overlay, hand_pos, r, color, -1)

        cv2.addWeighted(frame, 1-0.3, overlay, 0.3, 0, frame)
        return frame

    def render_neon(self, frame, hand):
        """Render neon outline effect"""
        landmarks = hand['landmarks'].astype(int)
        color = self.colors[self.palette][self.color_index]

        connections = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
                      (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                      (13,17),(17,18),(18,19),(19,20),(0,17)]

        for start, end in connections:
            p1 = tuple(landmarks[start])
            p2 = tuple(landmarks[end])
            cv2.line(frame, p1, p2, color, 5)
            cv2.line(frame, p1, p2, (255,255,255), 2)

        return frame

    def update(self, frame, hands, effect_mode):
        """Update all effects"""
        self.frame_count += 1

        for i, hand in enumerate(hands):
            pos = tuple(hand['palm_center'].astype(int))

            if effect_mode == 'particles':
                frame = self.render_particles(frame, pos)
            elif effect_mode == 'trails':
                frame = self.render_trail(frame, i, pos)
            elif effect_mode == 'glow':
                frame = self.render_glow(frame, pos)
            elif effect_mode == 'neon':
                frame = self.render_neon(frame, hand)

        if self.frame_count % 30 == 0:
            self.color_index = (self.color_index + 1) % len(self.colors[self.palette])

        return frame

# ============================================================================
# CLASS 4: MAIN APPLICATION
# ============================================================================

class ARHandEffectsApp:
    """Main application class"""

    def __init__(self):
        logger.info("Starting AR Hand Effects Generator")

        self.hand_detector = HandDetector()
        self.gesture_recognizer = GestureRecognizer()
        self.effects_engine = EffectsEngine()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        self.running = True
        self.effects_enabled = True
        self.show_landmarks = SHOW_HAND_LANDMARKS
        self.show_gestures = SHOW_GESTURES
        self.current_effect = 'particles'
        self.current_palette = 'neon'

        self.fps = 0
        self.frame_count = 0
        self.start_time = cv2.getTickCount()

    def handle_keyboard(self, key):
        """Handle keyboard input"""
        if key == ord('q'):
            self.running = False
            logger.info("Quit")
        elif key == ord('c'):
            _, frame = self.cap.read()
            filename = f"screenshots/screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, frame)
            logger.info(f"Screenshot saved: {filename}")
        elif key == ord(' '):
            self.effects_enabled = not self.effects_enabled
        elif key == ord('1'):
            self.current_effect = 'particles'
        elif key == ord('2'):
            self.current_effect = 'trails'
        elif key == ord('3'):
            self.current_effect = 'glow'
        elif key == ord('5'):
            self.current_effect = 'neon'
        elif key == ord('p'):
            palettes = ['neon', 'fire', 'ice', 'pastel']
            idx = palettes.index(self.current_palette)
            self.current_palette = palettes[(idx + 1) % len(palettes)]
            self.effects_engine.palette = self.current_palette
        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
        elif key == ord('g'):
            self.show_gestures = not self.show_gestures

    def run(self):
        """Main application loop"""
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                hands = self.hand_detector.detect(frame)

                if self.show_landmarks and hands:
                    frame = self.hand_detector.draw_landmarks(frame, hands)

                if self.effects_enabled and hands:
                    frame = self.effects_engine.update(frame, hands, self.current_effect)

                if self.show_gestures:
                    for hand in hands:
                        gesture = self.gesture_recognizer.recognize(hand)
                        if gesture:
                            x, y = int(hand['palm_center'][0]), int(hand['palm_center'][1])
                            cv2.putText(frame, f"{gesture} ({hand['handedness']})", (x-50, y-60),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                if SHOW_FPS:
                    cv2.putText(frame, f"FPS: {self.fps:.1f}", (10,30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                status = f"Hands: {len(hands)} | Effect: {self.current_effect.upper()} | Palette: {self.current_palette.upper()}"
                cv2.putText(frame, status, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                help_text = "Q:Quit C:Screenshot Space:Toggle 1-3,5:Effect P:Palette L:Landmarks G:Gestures"
                cv2.putText(frame, help_text, (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

                cv2.imshow("AR Hand Effects", frame)

                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self.handle_keyboard(key)

                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = (cv2.getTickCount() - self.start_time) / cv2.getTickFrequency()
                    self.fps = 30 / elapsed
                    self.start_time = cv2.getTickCount()

        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        self.cap.release()
        self.hand_detector.release()
        cv2.destroyAllWindows()
        logger.info("Done!")

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    app = ARHandEffectsApp()
    app.run()