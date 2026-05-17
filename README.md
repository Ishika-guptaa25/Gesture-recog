# AR Hand Effects Generator

**Real-time hand detection with visual effects 
## Features

Real-time hand detection (2 hands)
Visual effects (Particles, Trails, Glow, Neon)
Gesture recognition (Peace, Thumbs up, Pointing, Palm, Fist)
Color palettes (Neon, Fire, Ice, Pastel)
Screenshot capture (press C)

## Quick Start

```bash
# Clone repository
git clone https://github.com/Ishika-guptaa25/gesture-verse.git

# Go into project folder
cd gesture-verse

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Run project
python main.py
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| Q | Quit |
| C | Screenshot |
| Space | Toggle effects |
| 1 | Particles |
| 2 | Trails |
| 3 | Glow |
| 5 | Neon |
| P | Change palette |
| L | Landmarks on/off |
| G | Gestures on/off |

## Customization

Edit `main.py` lines 13-28 for settings:
- CAMERA_WIDTH/HEIGHT - Resolution
- TARGET_FPS - Frame rate
- MIN_DETECTION_CONFIDENCE - Hand sensitivity
- SHOW_HAND_LANDMARKS - Show skeleton
- EFFECT_INTENSITY - Effect strength

## Requirements

- Python 3.8+
- Webcam
- 4GB RAM
- macOS/Windows/Linux

## System Requirements

Minimum:
- Python 3.8+
- 4GB RAM
- Webcam

Recommended:
- Python 3.10+
- 8GB RAM
- 1080p Webcam
