"""
Effects engine for rendering visual effects on detected hands
Manages particles, trails, filters, and overlays
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class Particle:
    """Single particle for particle effects"""

    def __init__(self, x, y, vx, vy, color, lifetime=30):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.lifetime = lifetime
        self.initial_lifetime = lifetime
        self.size = 4

    def update(self):
        """Update particle position and lifetime"""
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        self.vx *= 0.95  # Friction
        self.vy *= 0.95

    def is_alive(self) -> bool:
        """Check if particle is still alive"""
        return self.lifetime > 0

    def get_alpha(self) -> float:
        """Get transparency based on remaining lifetime"""
        return self.lifetime / self.initial_lifetime


class EffectsEngine:
    """Main effects rendering engine"""

    def __init__(self):
        """Initialize effects engine"""
        self.particles = []
        self.trails = {}  # Dictionary of trails for each hand
        self.current_effect = 'particles'
        self.color_index = 0
        self.frame_count = 0

        # Colors
        self.colors = {
            'neon': [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)],
            'fire': [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0)],
            'ice': [(255, 128, 0), (0, 255, 255), (255, 255, 255), (0, 200, 255)],
            'pastel': [(255, 179, 198), (255, 223, 186), (255, 255, 200), (186, 255, 201)],
        }
        self.current_palette = 'neon'

        logger.info("EffectsEngine initialized")

    def render_particles(self, frame: np.ndarray, hand_position: Tuple[int, int]) -> np.ndarray:
        """
        Render particle burst effect at hand position

        Args:
            frame: Input frame
            hand_position: (x, y) position of hand

        Returns:
            Frame with particle effects
        """
        # Create particles on hand movement
        if self.frame_count % 3 == 0:  # Create particles every 3 frames
            for _ in range(5):
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.random.uniform(2, 6)
                vx = speed * np.cos(angle)
                vy = speed * np.sin(angle)

                color = self.colors[self.current_palette][
                    len(self.particles) % len(self.colors[self.current_palette])
                ]

                particle = Particle(
                    hand_position[0], hand_position[1],
                    vx, vy, color, lifetime=30
                )
                self.particles.append(particle)

        # Update and render particles
        for particle in self.particles[:]:
            particle.update()

            if not particle.is_alive():
                self.particles.remove(particle)
            else:
                x, y = int(particle.x), int(particle.y)
                size = int(particle.size * particle.get_alpha())

                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    cv2.circle(frame, (x, y), max(1, size), particle.color, -1)

        return frame

    def render_trail(self, frame: np.ndarray, hand_id: int,
                     hand_position: Tuple[int, int]) -> np.ndarray:
        """
        Render motion trail effect

        Args:
            frame: Input frame
            hand_id: Unique ID for this hand
            hand_position: (x, y) position of hand

        Returns:
            Frame with trail effect
        """
        if hand_id not in self.trails:
            self.trails[hand_id] = deque(maxlen=30)

        self.trails[hand_id].append(hand_position)

        # Draw trail
        trail_points = list(self.trails[hand_id])
        for i in range(len(trail_points) - 1):
            alpha = i / len(trail_points)
            color = self.colors[self.current_palette][i % len(self.colors[self.current_palette])]
            thickness = int(5 * alpha)

            if thickness > 0:
                cv2.line(frame, trail_points[i], trail_points[i + 1], color, thickness)

        return frame

    def render_glow(self, frame: np.ndarray, hand_position: Tuple[int, int],
                   radius: int = 50) -> np.ndarray:
        """
        Render glowing effect around hand

        Args:
            frame: Input frame
            hand_position: (x, y) position of hand
            radius: Glow radius in pixels

        Returns:
            Frame with glow effect
        """
        overlay = frame.copy()

        # Draw multiple circles with decreasing opacity
        color = self.colors[self.current_palette][self.color_index]

        for r in range(radius, 0, -5):
            alpha = (radius - r) / radius * 0.3
            cv2.circle(overlay, hand_position, r, color, -1)

        cv2.addWeighted(frame, 1 - 0.3, overlay, 0.3, 0, frame)

        return frame

    def render_pixelate(self, frame: np.ndarray, hand_position: Tuple[int, int],
                       region_size: int = 50, pixel_size: int = 5) -> np.ndarray:
        """
        Render pixelation effect around hand

        Args:
            frame: Input frame
            hand_position: (x, y) position of hand
            region_size: Size of region to pixelate
            pixel_size: Size of each pixel block

        Returns:
            Frame with pixelation effect
        """
        x, y = hand_position
        x1 = max(0, x - region_size // 2)
        y1 = max(0, y - region_size // 2)
        x2 = min(frame.shape[1], x + region_size // 2)
        y2 = min(frame.shape[0], y + region_size // 2)

        region = frame[y1:y2, x1:x2]

        # Resize down then up for pixelation
        small = cv2.resize(region, (region.shape[1] // pixel_size, region.shape[0] // pixel_size))
        pixelated = cv2.resize(small, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)

        frame[y1:y2, x1:x2] = pixelated

        return frame

    def render_neon_outline(self, frame: np.ndarray, hand_landmarks: np.ndarray) -> np.ndarray:
        """
        Render neon outline effect for hand

        Args:
            frame: Input frame
            hand_landmarks: Array of hand landmark positions

        Returns:
            Frame with neon outline
        """
        color = self.colors[self.current_palette][self.color_index]

        # Draw connections with glow
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]

        for start, end in connections:
            p1 = tuple(hand_landmarks[start].astype(int))
            p2 = tuple(hand_landmarks[end].astype(int))

            # Draw thick line for glow
            cv2.line(frame, p1, p2, color, 5)
            # Draw thin bright line
            cv2.line(frame, p1, p2, (255, 255, 255), 2)

        return frame

    def apply_color_filter(self, frame: np.ndarray, filter_type: str = 'rgb') -> np.ndarray:
        """
        Apply color filter to entire frame

        Args:
            frame: Input frame
            filter_type: Type of filter ('rgb', 'hsv', 'thermal', 'cool')

        Returns:
            Filtered frame
        """
        if filter_type == 'thermal':
            # Thermal effect using color map
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            colormap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            return cv2.addWeighted(frame, 0.3, colormap, 0.7, 0)

        elif filter_type == 'cool':
            # Cool blue tone
            b, g, r = cv2.split(frame)
            b = cv2.add(b, 50)
            return cv2.merge([b, g, r])

        elif filter_type == 'warm':
            # Warm red/yellow tone
            b, g, r = cv2.split(frame)
            r = cv2.add(r, 50)
            return cv2.merge([b, g, r])

        return frame

    def update(self, frame: np.ndarray, hands, effect_mode: str = 'particles') -> np.ndarray:
        """
        Update and render all effects

        Args:
            frame: Input frame
            hands: List of Hand objects
            effect_mode: Current effect mode

        Returns:
            Frame with rendered effects
        """
        self.frame_count += 1

        for i, hand in enumerate(hands):
            pos = tuple(hand.palm_center.astype(int))

            if effect_mode == 'particles':
                frame = self.render_particles(frame, pos)
            elif effect_mode == 'trails':
                frame = self.render_trail(frame, i, pos)
            elif effect_mode == 'glow':
                frame = self.render_glow(frame, pos)
            elif effect_mode == 'pixelate':
                frame = self.render_pixelate(frame, pos)
            elif effect_mode == 'neon':
                frame = self.render_neon_outline(frame, hand.landmarks)

        # Cycle through palette colors
        if self.frame_count % 30 == 0:
            self.color_index = (self.color_index + 1) % len(self.colors[self.current_palette])

        return frame

    def set_effect_mode(self, mode: str):
        """Set the current effect mode"""
        self.current_effect = mode
        logger.info(f"Effect mode changed to: {mode}")

    def set_palette(self, palette: str):
        """Set the current color palette"""
        if palette in self.colors:
            self.current_palette = palette
            logger.info(f"Palette changed to: {palette}")

    def clear_particles(self):
        """Clear all particles"""
        self.particles.clear()
        self.trails.clear()