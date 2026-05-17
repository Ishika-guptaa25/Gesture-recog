"""
Configuration settings for AR Hand Effects application
"""

# Camera Settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
TARGET_FPS = 30
CAMERA_INDEX = 0

# Hand Detection Settings
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2

# Effect Settings
EFFECT_INTENSITY = 0.8  # 0.0 - 1.0
PARTICLE_COUNT = 100
TRAIL_LENGTH = 30
COLOR_BLUR_KERNEL = 15

# Display Settings
SHOW_FPS = True
SHOW_HAND_LANDMARKS = True
SHOW_GESTURES = True
WINDOW_TITLE = "AR Hand Effects Generator"

# Effect Types
EFFECT_MODES = {
    'rainbow': {'description': 'Rainbow color cycling', 'intensity': 0.8},
    'particles': {'description': 'Particle burst effects', 'intensity': 0.9},
    'trails': {'description': 'Hand motion trails', 'intensity': 0.7},
    'pixelate': {'description': 'Pixelation effect', 'intensity': 0.6},
    'mirror': {'description': 'Mirror distortion', 'intensity': 0.75},
    'glow': {'description': 'Glowing hands', 'intensity': 0.8},
    'neon': {'description': 'Neon outline effect', 'intensity': 0.9},
}

# Gesture Thresholds
GESTURE_THRESHOLDS = {
    'peace_sign': 0.8,
    'thumbs_up': 0.75,
    'pointing': 0.8,
    'open_palm': 0.7,
    'fist': 0.75,
}

# Color Palettes
PALETTES = {
    'neon': [(255, 0, 127), (0, 255, 255), (255, 255, 0), (255, 0, 255)],
    'pastel': [(255, 179, 186), (255, 223, 186), (255, 250, 200), (186, 255, 201)],
    'dark': [(26, 188, 156), (52, 152, 219), (155, 89, 182), (231, 76, 60)],
    'fire': [(255, 0, 0), (255, 165, 0), (255, 255, 0), (255, 140, 0)],
}

# File Settings
SCREENSHOT_DIR = "screenshots"
LOG_FILE = "app.log"
CACHE_DIR = ".cache"

# Performance Settings
USE_GPU = True
MULTI_THREADING = True
MAX_WORKERS = 4

# Debug Settings
DEBUG_MODE = False
VERBOSE_LOGGING = False
PROFILE_PERFORMANCE = False