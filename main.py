"""
AR Hand Effects Generator - Main Application
Real-time hand detection with visual effects
"""

import cv2
import numpy as np
import sys
import logging
from typing import List

import config
from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from core.effects_engine import EffectsEngine
from utils.logger import setup_logger
from utils.file_handler import FileHandler

# Setup logging
logger = setup_logger(__name__, config.LOG_FILE)


class ARHandEffectsApp:
    """Main application class"""

    def __init__(self):
        """Initialize the application"""
        logger.info("=" * 50)
        logger.info("AR Hand Effects Generator - Initializing")
        logger.info("=" * 50)

        # Initialize components
        self.hand_detector = HandDetector(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            max_num_hands=config.MAX_NUM_HANDS
        )

        self.gesture_recognizer = GestureRecognizer()
        self.effects_engine = EffectsEngine()
        self.file_handler = FileHandler(
            screenshot_dir=config.SCREENSHOT_DIR,
            cache_dir=config.CACHE_DIR
        )

        # Application state
        self.running = True
        self.effects_enabled = True
        self.show_landmarks = config.SHOW_HAND_LANDMARKS
        self.show_gestures = config.SHOW_GESTURES
        self.current_effect = 'particles'
        self.current_palette = 'neon'

        # Camera setup
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = cv2.getTickCount()

        logger.info("Application initialized successfully")
        logger.info(f"Camera resolution: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
        logger.info(f"Target FPS: {config.TARGET_FPS}")

    def handle_keyboard(self, key: int):
        """Handle keyboard input"""
        if key == ord('q'):
            self.running = False
            logger.info("Quit command received")

        elif key == ord('c'):
            ret, frame = self.cap.read()
            if ret:
                filepath = self.file_handler.save_screenshot(frame)
                logger.info(f"Screenshot saved: {filepath}")

        elif key == ord(' '):
            self.effects_enabled = not self.effects_enabled
            logger.info(f"Effects toggled: {'ON' if self.effects_enabled else 'OFF'}")

        elif key == ord('l'):
            self.show_landmarks = not self.show_landmarks
            logger.info(f"Landmarks visibility: {'ON' if self.show_landmarks else 'OFF'}")

        elif key == ord('g'):
            self.show_gestures = not self.show_gestures
            logger.info(f"Gesture display: {'ON' if self.show_gestures else 'OFF'}")

        elif key == ord('1'):
            self.current_effect = 'particles'
            self.effects_engine.set_effect_mode('particles')
            logger.info("Effect: Particles")

        elif key == ord('2'):
            self.current_effect = 'trails'
            self.effects_engine.set_effect_mode('trails')
            logger.info("Effect: Trails")

        elif key == ord('3'):
            self.current_effect = 'glow'
            self.effects_engine.set_effect_mode('glow')
            logger.info("Effect: Glow")

        elif key == ord('4'):
            self.current_effect = 'pixelate'
            self.effects_engine.set_effect_mode('pixelate')
            logger.info("Effect: Pixelate")

        elif key == ord('5'):
            self.current_effect = 'neon'
            self.effects_engine.set_effect_mode('neon')
            logger.info("Effect: Neon")

        elif key == ord('p'):
            palettes = list(self.effects_engine.colors.keys())
            current_idx = palettes.index(self.current_palette)
            self.current_palette = palettes[(current_idx + 1) % len(palettes)]
            self.effects_engine.set_palette(self.current_palette)
            logger.info(f"Palette: {self.current_palette}")

    def draw_ui(self, frame: np.ndarray, hands: List, fps: float) -> np.ndarray:
        """Draw UI elements on frame"""
        height, width = frame.shape[:2]

        # FPS counter
        if config.SHOW_FPS:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Status info
        status_text = f"Hands: {len(hands)} | Effect: {self.current_effect.upper()} | Palette: {self.current_palette.upper()}"
        cv2.putText(frame, status_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Help text
        help_text = [
            "Q: Quit | C: Capture | Space: Toggle Effects | L: Landmarks | G: Gestures",
            "1-5: Change Effect | P: Palette | H: Help"
        ]

        for i, text in enumerate(help_text):
            cv2.putText(frame, text, (10, height - 40 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Gesture display
        if self.show_gestures and len(hands) > 0:
            for i, hand in enumerate(hands):
                gesture = self.gesture_recognizer.recognize(hand)
                if gesture:
                    x, y = int(hand.palm_center[0]), int(hand.palm_center[1])
                    cv2.putText(frame, f"{gesture} ({hand.handedness})",
                                (x - 50, y - 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)

        return frame

    def run(self):
        """Main application loop"""
        logger.info("Starting main loop")

        try:
            while self.running:
                ret, frame = self.cap.read()

                if not ret:
                    logger.error("Failed to read frame from camera")
                    break

                # Flip frame for selfie view
                frame = cv2.flip(frame, 1)

                # Detect hands
                hands = self.hand_detector.detect(frame)

                # Draw landmarks
                if self.show_landmarks and len(hands) > 0:
                    frame = self.hand_detector.draw_landmarks(frame, hands)

                # Apply effects
                if self.effects_enabled and len(hands) > 0:
                    frame = self.effects_engine.update(frame, hands, self.current_effect)

                # Draw UI
                frame = self.draw_ui(frame, hands, self.fps)

                # Display frame
                cv2.imshow(config.WINDOW_TITLE, frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    self.handle_keyboard(key)

                # Update FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = (cv2.getTickCount() - self.start_time) / cv2.getTickFrequency()
                    self.fps = 30 / elapsed
                    self.start_time = cv2.getTickCount()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources")

        if self.cap:
            self.cap.release()

        self.hand_detector.release()
        cv2.destroyAllWindows()

        # Print summary
        logger.info("=" * 50)
        logger.info(f"Total frames processed: {self.frame_count}")
        logger.info(f"Screenshots taken: {self.file_handler.get_screenshot_count()}")
        logger.info("Application closed successfully")
        logger.info("=" * 50)


def main():
    """Main entry point"""
    try:
        app = ARHandEffectsApp()
        app.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()