# Calibration Pattern
# Displays ternary-encoded LED IDs for camera-based 3D position calibration
#
# Each LED displays a unique color sequence encoding its ID in base-3 (ternary).
# Pattern sequence (each frame held for 0.2 seconds):
#   Frame 0: Black
#   Frame 1: Magenta (sync marker)
#   Frame 2: Black
#   Frames 3-20: 9 ternary digit frames, each followed by black frame
#     Frame 3: digit 0 color, Frame 4: Black
#     Frame 5: digit 1 color, Frame 6: Black
#     ...
#     Frame 19: digit 8 color, Frame 20: Black
#
# Ternary encoding:
#   - Red = 0
#   - Green = 1
#   - Blue = 2
#
# LED ID encoding (with checksum):
#   1. Multiply LED ID by 9
#   2. Add checksum to make value divisible by 7
#   3. Convert to base-3 representation (LSB first)

import micropython
from animation_base import Animation


class CalibrationAnimation(Animation):
    """
    Camera calibration pattern using ternary color encoding.
    """

    def __init__(self, coords, current_time):
        super().__init__(coords, current_time)

        # Timing parameters
        self.state_duration = 0.2  # Duration of each state in seconds
        self.num_ternary_digits = 9
        self.total_frames = 3 + (self.num_ternary_digits * 2)  # 21 frames total

        # Ternary digit colors (R, G, B)
        self.ternary_colors = (
            (1.0, 0.0, 0.0),  # 0: Red
            (0.0, 1.0, 0.0),  # 1: Green
            (0.0, 0.0, 1.0),  # 2: Blue
        )

        # Sync marker color
        self.magenta = (1.0, 0.0, 1.0)

        # Black color
        self.black = (0.0, 0.0, 0.0)

    @micropython.native
    def get_ternary_digit(self, led_id, digit_pos):
        """
        Encode LED ID into ternary with checksum.
        Returns the ternary digit at the specified position (0 = least significant).

        Algorithm:
          n = led_id * 9  # Make space for checksum
          n = n + (7 - (n % 7))  # Add checksum to make divisible by 7
          Convert to base-3 representation
        """
        n = led_id * 9
        n = n + (7 - (n % 7))

        # Extract the digit at digit_pos by repeatedly dividing by 3
        for _ in range(digit_pos):
            n = n // 3

        return n % 3

    @micropython.native
    def get_color(self, x, y, z, t, led_id):
        """
        Calculate color based on calibration pattern timing.
        """
        # Calculate which frame we're in based on time (pattern cycles every 4.2 seconds)
        # Each state lasts state_duration seconds (0.2s by default)
        cycle_frame = int(t / self.state_duration) % self.total_frames

        # Determine color based on cycle frame
        if cycle_frame == 0 or cycle_frame == 2:
            # Black frame (frame 0 and 2)
            return self.black
        elif cycle_frame == 1:
            # Magenta sync marker
            return self.magenta
        else:
            # Digit frames (3-20)
            digit_frame = cycle_frame - 3

            if digit_frame % 2 == 0:
                # Even frames (0, 2, 4, ..., 16) show digit
                digit = self.get_ternary_digit(led_id, digit_frame // 2)
                return self.ternary_colors[digit]
            else:
                # Odd frames show black (spacing between digits)
                return self.black
