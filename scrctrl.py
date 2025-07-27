import cv2
import numpy as np
import subprocess
import time
import threading
import queue
import logging
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

opencv_to_adb_map_str = {
    # --- Standard Keys (usually consistent across systems if masked to ASCII) ---
    13: "KEYCODE_ENTER",   # Enter (ASCII CR)
    27: "KEYCODE_BACK",    # Esc (ASCII ESC) -> common Android 'back' action
    8: "KEYCODE_DEL",     # Backspace / Delete (ASCII DEL)
    9: "KEYCODE_TAB",      # Tab (ASCII TAB)

    # --- Printable ASCII Characters (mapped to their respective KEYCODE_ strings) ---
    # It's more efficient to generate these dynamically or use a helper for `input text`,
    # but for a direct KEYCODE_ string return, we'll list them.
    # Note: ADB often uses a general KEYCODE for a character regardless of case (A vs a).
    # You might want to map both 'a' and 'A' to "KEYCODE_A"
    ord('0'): "KEYCODE_0",
    ord('1'): "KEYCODE_1",
    ord('2'): "KEYCODE_2",
    ord('3'): "KEYCODE_3",
    ord('4'): "KEYCODE_4",
    ord('5'): "KEYCODE_5",
    ord('6'): "KEYCODE_6",
    ord('7'): "KEYCODE_7",
    ord('8'): "KEYCODE_8",
    ord('9'): "KEYCODE_9",

    ord('a'): "KEYCODE_A", ord('A'): "KEYCODE_A",
    ord('b'): "KEYCODE_B", ord('B'): "KEYCODE_B",
    ord('c'): "KEYCODE_C", ord('C'): "KEYCODE_C",
    ord('d'): "KEYCODE_D", ord('D'): "KEYCODE_D",
    ord('e'): "KEYCODE_E", ord('E'): "KEYCODE_E",
    ord('f'): "KEYCODE_F", ord('F'): "KEYCODE_F",
    ord('g'): "KEYCODE_G", ord('G'): "KEYCODE_G",
    ord('h'): "KEYCODE_H", ord('H'): "KEYCODE_H",
    ord('i'): "KEYCODE_I", ord('I'): "KEYCODE_I",
    ord('j'): "KEYCODE_J", ord('J'): "KEYCODE_J",
    ord('k'): "KEYCODE_K", ord('K'): "KEYCODE_K",
    ord('l'): "KEYCODE_L", ord('L'): "KEYCODE_L",
    ord('m'): "KEYCODE_M", ord('M'): "KEYCODE_M",
    ord('n'): "KEYCODE_N", ord('N'): "KEYCODE_N",
    ord('o'): "KEYCODE_O", ord('O'): "KEYCODE_O",
    ord('p'): "KEYCODE_P", ord('P'): "KEYCODE_P",
    ord('q'): "KEYCODE_Q", ord('Q'): "KEYCODE_Q",
    ord('r'): "KEYCODE_R", ord('R'): "KEYCODE_R",
    ord('s'): "KEYCODE_S", ord('S'): "KEYCODE_S",
    ord('t'): "KEYCODE_T", ord('T'): "KEYCODE_T",
    ord('u'): "KEYCODE_U", ord('U'): "KEYCODE_U",
    ord('v'): "KEYCODE_V", ord('V'): "KEYCODE_V",
    ord('w'): "KEYCODE_W", ord('W'): "KEYCODE_W",
    ord('x'): "KEYCODE_X", ord('X'): "KEYCODE_X",
    ord('y'): "KEYCODE_Y", ord('Y'): "KEYCODE_Y",
    ord('z'): "KEYCODE_Z", ord('Z'): "KEYCODE_Z",

    ord(' '): "KEYCODE_SPACE",
    ord('-'): "KEYCODE_MINUS",
    ord('='): "KEYCODE_EQUALS",
    ord('['): "KEYCODE_LEFT_BRACKET",
    ord(']'): "KEYCODE_RIGHT_BRACKET",
    ord('\\'): "KEYCODE_BACKSLASH",
    ord(';'): "KEYCODE_SEMICOLON",
    ord('\''): "KEYCODE_APOSTROPHE",
    ord('/'): "KEYCODE_SLASH",
    ord('.'): "KEYCODE_PERIOD",
    ord(','): "KEYCODE_COMMA",
    ord('`'): "KEYCODE_GRAVE",

    # --- Special Keys (THESE ARE EXAMPLES - YOU MUST REPLACE WITH YOUR SYSTEM'S VALUES!) ---
    # These are typical values for Windows/some Linux, but they WILL vary.
    # Run a key discovery script to get your specific values.
    # Format: OpenCV_Raw_Keycode: ADB_Keycode_String

    # Arrow Keys (e.g., from Windows/some Linux)
    2490368: "KEYCODE_DPAD_UP",
    2621440: "KEYCODE_DPAD_DOWN",
    2424832: "KEYCODE_DPAD_LEFT",
    2555904: "KEYCODE_DPAD_RIGHT",

    # Home, End, Page Up/Down, Insert (e.g., from Windows/some Linux)
    2359296: "KEYCODE_MOVE_HOME", # PC Home key
    2359552: "KEYCODE_MOVE_END",  # PC End key
    2162688: "KEYCODE_PAGE_UP",
    2228224: "KEYCODE_PAGE_DOWN",
    2293760: "KEYCODE_INSERT",

    # Function Keys (F1-F12) - VERY SYSTEM DEPENDENT VALUES!
    # Example values (replace with your discovered values!):
    # 65436: "KEYCODE_F1",
    # 65437: "KEYCODE_F2",
    # 65438: "KEYCODE_F3",
    # 65439: "KEYCODE_F4",
    # 65440: "KEYCODE_F5",
    # 65441: "KEYCODE_F6",
    # 65442: "KEYCODE_F7",
    # 65443: "KEYCODE_F8",
    # 65444: "KEYCODE_F9",
    # 65445: "KEYCODE_F10",
    # 65446: "KEYCODE_F11",
    # 65447: "KEYCODE_F12",

    # Numpad Keys - Also highly system dependent values!
    # Example (replace with your discovered values!):
    # 1048624: "KEYCODE_NUMPAD_0",
    # 1048625: "KEYCODE_NUMPAD_1",
    # # ... fill in for 2-9
    # 1048650: "KEYCODE_NUMPAD_ADD",
    # 1048649: "KEYCODE_NUMPAD_SUBTRACT",
    # 1048648: "KEYCODE_NUMPAD_MULTIPLY",
    # 1048647: "KEYCODE_NUMPAD_DIVIDE",
    # 1048651: "KEYCODE_NUMPAD_DOT",

    # Common Android System Keycodes (no direct PC key, but useful for mapping)
    # You could map 'h' to home, 'r' to recent apps, etc.
    # ord('h'): "KEYCODE_HOME",
    # ord('r'): "KEYCODE_APP_SWITCH", # Recent Apps
    # ord('v'): "KEYCODE_VOLUME_UP",
    # ord('w'): "KEYCODE_VOLUME_DOWN",
    # ord('p'): "KEYCODE_POWER",
    # ord('m'): "KEYCODE_MENU",
    # ord('z'): "KEYCODE_CAMERA",
    # ord('f'): "KEYCODE_BRIGHTNESS_DOWN",
    # ord('g'): "KEYCODE_BRIGHTNESS_UP",
}

def convert_opencv_to_adb_keycode_string(opencv_key_raw: int) -> str:
    """
    Converts an OpenCV raw key code to its corresponding ADB Keycode string.

    Args:
        opencv_key_raw: The raw integer value returned by cv2.waitKey() or cv2.waitKeyEx().
                        (e.g., from an OpenCV window capturing keyboard input).

    Returns:
        str: The ADB Keycode string (e.g., "KEYCODE_ENTER", "KEYCODE_A").
             Returns "KEYCODE_UNKNOWN" if no direct mapping is found.
    """
    key_masked = opencv_key_raw & 0xFF # Mask to get common ASCII/base values

    # 1. Check for special keys defined in our custom map (using raw value first)
    if opencv_key_raw in opencv_to_adb_map_str:
        return opencv_to_adb_map_str[opencv_key_raw]
    # 2. If not found by raw value, check for printable ASCII characters via masked value
    #    This covers most alphanumeric characters and common symbols.
    elif 32 <= key_masked <= 126: # Printable ASCII range
        char_val = key_masked
        # For characters, convert to uppercase and prepend "KEYCODE_"
        # ADB keycodes like KEYCODE_A handle both 'a' and 'A'.
        return f"KEYCODE_{chr(char_val).upper()}"
    # 3. Fallback: Check if the masked value itself is directly in the map
    #    (This might catch some special keys if they happen to have simple masked values)
    elif key_masked in opencv_to_adb_map_str:
        return opencv_to_adb_map_str[key_masked]
    else:
        # Unhandled key
        return "KEYCODE_UNKNOWN"



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('watch_controller.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GestureType(Enum):
    """Enumeration of supported gesture types."""
    TAP = "tap"
    SWIPE = "swipe"
    LONG_PRESS = "long_press"
    SCROLL = "scroll"


@dataclass
class DisplayConfig:
    """Configuration for display settings."""
    watch_width: int = 480
    watch_height: int = 480
    scale: float = 1.0
    button_width: int = 80
    button_height: int = 50
    button_spacing: int = 10
    
    @property
    def window_width(self) -> int:
        return int(self.watch_width * self.scale)
    
    @property
    def window_height(self) -> int:
        return int(self.watch_height * self.scale)
    
    @property
    def ui_width(self) -> int:
        return self.window_width + self.button_width + self.button_spacing * 2
    
    @property
    def ui_height(self) -> int:
        return self.window_height


@dataclass
class Button:
    """Represents a UI control button."""
    name: str
    keycode: int
    color: Tuple[int, int, int] = (50, 50, 50)
    text_color: Tuple[int, int, int] = (255, 255, 255)


class StreamConfig:
    """Configuration for video stream settings."""
    def __init__(self):
        self.bitrate = "16m"
        self.buffer_size = 10**8
        self.ffmpeg_flags = [
            '-loglevel', 'quiet',
            '-probesize', '32',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-fflags', '+genpts'
        ]


class AdbInterface:
    """Handles all ADB communication with proper error handling."""
    
    @staticmethod
    def check_device_connected() -> bool:
        """Check if an ADB device is connected."""
        try:
            result = subprocess.run(
                ['adb', 'devices'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            lines = result.stdout.strip().split('\n')
            connected_devices = [line for line in lines[1:] if 'device' in line and 'offline' not in line]
            return len(connected_devices) > 0
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"ADB check failed: {e}")
            return False
    
    @staticmethod
    def execute_command(*args: str) -> bool:
        """Execute an ADB command asynchronously."""
        try:
            command = ['adb', 'shell'] + list(args)
            subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            return True
        except Exception as e:
            logger.error(f"ADB command failed: {e}")
            return False
    
    @staticmethod
    def tap(x: int, y: int) -> bool:
        """Send tap command to device."""
        logger.debug(f"Tap at ({x}, {y})")
        return AdbInterface.execute_command('input', 'tap', str(x), str(y))
    
    @staticmethod
    def swipe(x1: int, y1: int, x2: int, y2: int, duration: int = 300) -> bool:
        """Send swipe command to device."""
        logger.debug(f"Swipe from ({x1}, {y1}) to ({x2}, {y2})")
        return AdbInterface.execute_command('input', 'swipe', str(x1), str(y1), str(x2), str(y2), str(duration))
    
    @staticmethod
    def long_press(x: int, y: int, duration: int = 800) -> bool:
        """Send long press command to device."""
        logger.debug(f"Long press at ({x}, {y})")
        return AdbInterface.swipe(x, y, x, y, duration)
    
    @staticmethod
    def key_event(keycode: int) -> bool:
        """Send key event to device."""
        logger.debug(f"Key event: {keycode}")
        return AdbInterface.execute_command('input', 'keyevent', str(keycode))
    


class VideoStream:
    """Manages video stream capture and decoding with robust frame handling."""
    
    def __init__(self, config: DisplayConfig, stream_config: StreamConfig):
        self.config = config
        self.stream_config = stream_config
        self.adb_process: Optional[subprocess.Popen] = None
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.is_running = False
        self._lock = threading.Lock()
        self.frame_size = self.config.watch_width * self.config.watch_height * 3
        self.buffer = bytearray()
        self.consecutive_failures = 0
        self.max_failures = 5
        self.read_timeout = 0.1
    
    def start(self) -> bool:
        """Start the video stream with improved parameters."""
        with self._lock:
            if self.is_running:
                return True
            
            try:
                logger.info("Starting video stream...")
                
                # Start ADB screen recording with optimized settings
                self.adb_process = subprocess.Popen([
                    'adb', 'exec-out', 'screenrecord',
                    '--bit-rate=8m',  # Reduced bitrate for stability
                    f'--size={self.config.watch_width}x{self.config.watch_height}',
                    '--output-format=h264',
                    '--time-limit=3600',  # 1 hour limit to prevent infinite recording
                    '-'
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                   bufsize=0)  # Unbuffered for immediate data
                
                # Start FFmpeg decoder with better synchronization
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-loglevel', 'error',  # Only show errors
                    '-fflags', '+genpts+igndts',  # Generate timestamps, ignore DTS
                    '-flags', 'low_delay',
                    '-probesize', '1024',  # Smaller probe size
                    '-analyzeduration', '1000000',  # 1 second analysis
                    '-max_delay', '0',  # No delay
                    '-i', 'pipe:0',
                    '-f', 'rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', f'{self.config.watch_width}x{self.config.watch_height}',
                    '-vsync', '0',  # No frame rate conversion
                    'pipe:1'
                ]
                
                self.ffmpeg_process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=self.adb_process.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0  # Unbuffered
                )
                
                self.adb_process.stdout.close()
                self.buffer.clear()
                self.consecutive_failures = 0
                self.is_running = True
                logger.info("Video stream started successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start video stream: {e}")
                self.stop()
                return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """Read a frame with robust buffering and error recovery."""
        if not self.is_running or not self.ffmpeg_process:
            return None
        
        try:
            # Read data in chunks and buffer it
            while len(self.buffer) < self.frame_size:
                try:
                    # Read available data (non-blocking approach)
                    chunk = self.ffmpeg_process.stdout.read(8192)  # 8KB chunks
                    if not chunk:
                        # Process died or no more data
                        if self.ffmpeg_process.poll() is not None:
                            logger.warning("FFmpeg process terminated")
                            self.consecutive_failures += 1
                            return None
                        continue
                    
                    self.buffer.extend(chunk)
                    
                except Exception as e:
                    logger.error(f"Error reading chunk: {e}")
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_failures:
                        logger.error("Too many consecutive failures, stopping stream")
                        self.is_running = False
                    return None
            
            # Extract one frame from buffer
            frame_data = bytes(self.buffer[:self.frame_size])
            self.buffer = self.buffer[self.frame_size:]
            
            # Convert to numpy array
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                (self.config.watch_height, self.config.watch_width, 3)
            )
            
            # Reset failure counter on success
            self.consecutive_failures = 0
            return frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            self.consecutive_failures += 1
            # Clear corrupted buffer
            self.buffer.clear()
            return None
    
    def get_buffer_info(self) -> dict:
        """Get information about the current buffer state."""
        return {
            'buffer_size': len(self.buffer),
            'frames_buffered': len(self.buffer) // self.frame_size,
            'consecutive_failures': self.consecutive_failures,
            'is_running': self.is_running
        }
    
    def stop(self):
        """Stop the video stream."""
        with self._lock:
            self.is_running = False
            logger.info("Stopping video stream...")
            
            # Clear buffer
            self.buffer.clear()
            
            for process_name, process in [('ffmpeg', self.ffmpeg_process), ('adb', self.adb_process)]:
                if process:
                    try:
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Force killing {process_name} process")
                            process.kill()
                            process.wait()
                    except Exception as e:
                        logger.error(f"Error stopping {process_name} process: {e}")
            
            self.adb_process = None
            self.ffmpeg_process = None


class FrameReader(threading.Thread):
    """Dedicated thread for reading frames with improved synchronization."""
    
    def __init__(self, stream: VideoStream, frame_queue: queue.Queue):
        super().__init__(daemon=True, name="FrameReader")
        self.stream = stream
        self.frame_queue = frame_queue
        self.running = True
        self._stats = {
            'frames_read': 0,
            'frames_dropped': 0,
            'incomplete_frames': 0,
            'last_reset': time.time(),
            'last_successful_frame': time.time()
        }
        self.frame_timeout = 5.0  # 5 seconds without frames = problem
        self.stats_lock = threading.Lock()
    
    def run(self):
        """Main thread loop with improved frame handling."""
        logger.info("Frame reader thread started")
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.running and self.stream.is_running:
            try:
                frame = self.stream.read_frame()
                
                if frame is None:
                    consecutive_failures += 1
                    
                    # Check if we've been without frames for too long
                    time_since_last_frame = time.time() - self._stats['last_successful_frame']
                    if time_since_last_frame > self.frame_timeout:
                        logger.warning(f"No frames received for {time_since_last_frame:.1f} seconds")
                        # Signal that stream needs restart
                        self.running = False
                        break
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive frame read failures")
                        self.running = False
                        break
                    
                    # Small delay to prevent busy waiting
                    time.sleep(0.01)
                    continue
                
                # Successfully got a frame
                consecutive_failures = 0
                with self.stats_lock:
                    self._stats['frames_read'] += 1
                    self._stats['last_successful_frame'] = time.time()
                
                # Drop old frames to maintain low latency
                dropped_frames = 0
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                        dropped_frames += 1
                    except queue.Empty:
                        break
                
                if dropped_frames > 0:
                    with self.stats_lock:
                        self._stats['frames_dropped'] += dropped_frames
                
                # Add new frame
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # This shouldn't happen with our dropping logic, but just in case
                    with self.stats_lock:
                        self._stats['frames_dropped'] += 1
                
            except Exception as e:
                logger.error(f"Unexpected error in frame reader: {e}")
                consecutive_failures += 1
                time.sleep(0.1)  # Longer delay after exceptions
        
        logger.info("Frame reader thread stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get frame reading statistics with thread safety."""
        with self.stats_lock:
            current_time = time.time()
            elapsed = max(current_time - self._stats['last_reset'], 0.1)  # Prevent division by zero
            
            # Get buffer info from stream
            buffer_info = self.stream.get_buffer_info()
            
            return {
                'fps': self._stats['frames_read'] / elapsed,
                'frames_read': self._stats['frames_read'],
                'frames_dropped': self._stats['frames_dropped'],
                'incomplete_frames': self._stats['incomplete_frames'],
                'drop_rate': self._stats['frames_dropped'] / max(self._stats['frames_read'], 1),
                'buffer_size_kb': buffer_info['buffer_size'] / 1024,
                'frames_buffered': buffer_info['frames_buffered'],
                'consecutive_failures': buffer_info['consecutive_failures'],
                'time_since_last_frame': current_time - self._stats['last_successful_frame']
            }
    
    def reset_stats(self):
        """Reset statistics counters."""
        with self.stats_lock:
            self._stats.update({
                'frames_read': 0,
                'frames_dropped': 0,
                'incomplete_frames': 0,
                'last_reset': time.time()
            })
    
    def stop(self):
        """Stop the frame reader thread."""
        self.running = False


class GestureRecognizer:
    """Enhanced gesture recognition with better accuracy."""
    
    def __init__(self, config: DisplayConfig):
        self.config = config
        self.state = {
            'is_dragging': False,
            'drag_start': None,
            'button_pressed': None,
            'start_time': None,
            'gesture_triggered': False,
            'last_position': None
        }
        self.thresholds = {
            'long_press_duration': 0.6,
            'movement_threshold': 8,
            'swipe_min_distance': 15,
            'scroll_min_distance': 10
        }
    
    def window_to_device_coords(self, x: int, y: int) -> Tuple[Optional[int], Optional[int]]:
        """Convert window coordinates to device coordinates."""
        if x > self.config.window_width:
            return None, None
        
        device_x = int(x * self.config.watch_width / self.config.window_width)
        device_y = int(y * self.config.watch_height / self.config.window_height)
        
        # Clamp to device bounds
        device_x = max(0, min(device_x, self.config.watch_width - 1))
        device_y = max(0, min(device_y, self.config.watch_height - 1))
        
        return device_x, device_y
    
    def get_button_at_position(self, x: int, y: int, buttons: list) -> Optional[int]:
        """Get button index at given position."""
        for idx in range(len(buttons)):
            bx = self.config.window_width + self.config.button_spacing
            by = idx * (self.config.button_height + self.config.button_spacing) + self.config.button_spacing
            
            if (bx <= x <= bx + self.config.button_width and 
                by <= y <= by + self.config.button_height):
                return idx
        return None
    
    def process_mouse_event(self, event: int, x: int, y: int, buttons: list) -> Optional[Dict[str, Any]]:
        """Process mouse event and return gesture information."""
        current_time = time.time()
        
        if event == cv2.EVENT_LBUTTONDOWN:
            return self._handle_left_button_down(x, y, current_time, buttons)
        elif event == cv2.EVENT_MOUSEMOVE:
            return self._handle_mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            return self._handle_left_button_up(x, y, current_time, buttons)
        elif event == cv2.EVENT_RBUTTONDOWN:
            return self._handle_right_button_down(x, y, current_time)
        elif event == cv2.EVENT_RBUTTONUP:
            return self._handle_right_button_up(x, y)
        
        return None
    
    def _handle_left_button_down(self, x: int, y: int, current_time: float, buttons: list) -> Dict[str, Any]:
        """Handle left mouse button down."""
        self.state.update({
            'is_dragging': True,
            'drag_start': (x, y),
            'start_time': current_time,
            'gesture_triggered': False,
            'last_position': (x, y),
            'button_pressed': self.get_button_at_position(x, y, buttons)
        })
        return {'type': 'button_down', 'position': (x, y)}
    
    def _handle_mouse_move(self, x: int, y: int) -> Optional[Dict[str, Any]]:
        """Handle mouse movement."""
        if not self.state['is_dragging']:
            return None
        
        if self.state['last_position']:
            dx = abs(x - self.state['drag_start'][0])
            dy = abs(y - self.state['drag_start'][1])
            
            if dx > self.thresholds['movement_threshold'] or dy > self.thresholds['movement_threshold']:
                self.state['gesture_triggered'] = True
        
        self.state['last_position'] = (x, y)
        return {'type': 'move', 'position': (x, y)}
    
    def _handle_left_button_up(self, x: int, y: int, current_time: float, buttons: list) -> Optional[Dict[str, Any]]:
        """Handle left mouse button up."""
        if not self.state['is_dragging']:
            return None
        
        start_x, start_y = self.state['drag_start']
        duration = current_time - self.state['start_time']
        distance = ((x - start_x) ** 2 + (y - start_y) ** 2) ** 0.5
        
        self.state['is_dragging'] = False
        
        # Handle button interactions
        if self.state['button_pressed'] is not None:
            button_idx = self.get_button_at_position(x, y, buttons)
            if button_idx == self.state['button_pressed']:
                gesture_type = GestureType.LONG_PRESS if (duration >= self.thresholds['long_press_duration'] and 
                                                         not self.state['gesture_triggered']) else GestureType.TAP
                return {
                    'type': 'button_gesture',
                    'gesture': gesture_type,
                    'button_index': button_idx,
                    'duration': duration
                }
        
        # Handle screen interactions
        else:
            device_start = self.window_to_device_coords(start_x, start_y)
            device_end = self.window_to_device_coords(x, y)
            
            if device_start[0] is None or device_end[0] is None:
                return None
            
            if duration >= self.thresholds['long_press_duration'] and not self.state['gesture_triggered']:
                return {
                    'type': 'screen_gesture',
                    'gesture': GestureType.LONG_PRESS,
                    'position': device_start,
                    'duration': duration
                }
            elif distance < self.thresholds['movement_threshold']:
                return {
                    'type': 'screen_gesture',
                    'gesture': GestureType.TAP,
                    'position': device_start
                }
            elif distance >= self.thresholds['swipe_min_distance']:
                return {
                    'type': 'screen_gesture',
                    'gesture': GestureType.SWIPE,
                    'start': device_start,
                    'end': device_end,
                    'distance': distance
                }
        
        return None
    
    def _handle_right_button_down(self, x: int, y: int, current_time: float) -> Dict[str, Any]:
        """Handle right mouse button down for scrolling."""
        if x <= self.config.window_width:
            self.state.update({
                'is_dragging': True,
                'drag_start': (x, y),
                'start_time': current_time,
                'button_pressed': 'right'
            })
        return {'type': 'scroll_start', 'position': (x, y)}
    
    def _handle_right_button_up(self, x: int, y: int) -> Optional[Dict[str, Any]]:
        """Handle right mouse button up for scrolling."""
        if (self.state['is_dragging'] and self.state['button_pressed'] == 'right' and 
            self.state['drag_start']):
            
            start_x, start_y = self.state['drag_start']
            device_start = self.window_to_device_coords(start_x, start_y)
            device_end = self.window_to_device_coords(x, y)
            
            self.state['is_dragging'] = False
            self.state['button_pressed'] = None
            
            if (device_start[0] is not None and device_end[0] is not None and 
                abs(device_end[1] - device_start[1]) > self.thresholds['scroll_min_distance']):
                return {
                    'type': 'screen_gesture',
                    'gesture': GestureType.SCROLL,
                    'start': device_start,
                    'end': device_end
                }
        
        return None


class UIRenderer:
    """Handles all UI rendering with improved visual feedback."""
    
    def __init__(self, config: DisplayConfig, buttons: list):
        self.config = config
        self.buttons = buttons
        self.colors = {
            'background': (20, 20, 20),
            'button_normal': (50, 50, 50),
            'button_hover': (70, 70, 70),
            'button_pressed': (90, 90, 90),
            'text': (255, 255, 255),
            'fps_good': (0, 255, 0),
            'fps_warning': (255, 255, 0),
            'fps_bad': (255, 0, 0)
        }
    
    def render_frame(self, frame: np.ndarray, fps: float, stats: Optional[Dict] = None) -> np.ndarray:
        """Render the complete UI frame."""
        # Create UI background
        ui_frame = np.full((self.config.ui_height, self.config.ui_width, 3), 
                          self.colors['background'], dtype=np.uint8)
        
        # Render watch screen
        if frame is not None:
            resized_frame = cv2.resize(frame, (self.config.window_width, self.config.window_height))
            ui_frame[0:self.config.window_height, 0:self.config.window_width] = resized_frame
        
        # Render control buttons
        self._render_buttons(ui_frame)
        
        # Render status information
        self._render_status(ui_frame, fps, stats)
        
        return ui_frame
    
    def _render_buttons(self, ui_frame: np.ndarray):
        """Render control buttons."""
        for idx, button in enumerate(self.buttons):
            bx = self.config.window_width + self.config.button_spacing
            by = idx * (self.config.button_height + self.config.button_spacing) + self.config.button_spacing
            
            # Button background
            cv2.rectangle(ui_frame, (bx, by), 
                         (bx + self.config.button_width, by + self.config.button_height),
                         button.color, -1)
            
            # Button border
            cv2.rectangle(ui_frame, (bx, by), 
                         (bx + self.config.button_width, by + self.config.button_height),
                         (100, 100, 100), 2)
            
            # Button text
            text_size = cv2.getTextSize(button.name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = bx + (self.config.button_width - text_size[0]) // 2
            text_y = by + (self.config.button_height + text_size[1]) // 2
            
            cv2.putText(ui_frame, button.name, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, button.text_color, 2)
    
    def _render_status(self, ui_frame: np.ndarray, fps: float, stats: Optional[Dict]):
        """Render enhanced status information including buffer status."""
        y_pos = 25
        line_height = 20
        
        # FPS indicator with more granular color coding
        if fps > 25:
            fps_color = self.colors['fps_good']
        elif fps > 15:
            fps_color = self.colors['fps_warning'] 
        else:
            fps_color = self.colors['fps_bad']
            
        cv2.putText(ui_frame, f"FPS: {fps:.1f}", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        y_pos += line_height
        
        # Stream health indicators
        if stats:
            # Buffer status
            buffer_size = stats.get('buffer_size_kb', 0)
            buffer_color = self.colors['fps_good'] if buffer_size < 100 else (
                self.colors['fps_warning'] if buffer_size < 500 else self.colors['fps_bad']
            )
            cv2.putText(ui_frame, f"Buffer: {buffer_size:.1f}KB", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, buffer_color, 1)
            y_pos += 15
            
            # Drop rate
            drop_rate = stats.get('drop_rate', 0) * 100
            drop_color = self.colors['fps_good'] if drop_rate < 5 else (
                self.colors['fps_warning'] if drop_rate < 15 else self.colors['fps_bad']
            )
            cv2.putText(ui_frame, f"Drop: {drop_rate:.1f}%", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, drop_color, 1)
            y_pos += 15
            
            # Time since last frame
            time_since_frame = stats.get('time_since_last_frame', 0)
            if time_since_frame > 1.0:
                time_color = self.colors['fps_bad']
                cv2.putText(ui_frame, f"Stale: {time_since_frame:.1f}s", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, time_color, 1)
                y_pos += 15
            
            # Consecutive failures warning
            failures = stats.get('consecutive_failures', 0)
            if failures > 0:
                fail_color = self.colors['fps_warning'] if failures < 3 else self.colors['fps_bad']
                cv2.putText(ui_frame, f"Fails: {failures}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, fail_color, 1)
        
        # Connection status indicator (top right)
        status_x = self.config.window_width - 80
        if stats and stats.get('time_since_last_frame', 0) < 1.0:
            cv2.circle(ui_frame, (status_x, 15), 8, self.colors['fps_good'], -1)
            cv2.putText(ui_frame, "LIVE", (status_x - 25, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['fps_good'], 1)
        else:
            cv2.circle(ui_frame, (status_x, 15), 8, self.colors['fps_bad'], -1)
            cv2.putText(ui_frame, "STALE", (status_x - 30, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['fps_bad'], 1)


class WatchController:
    """Main application controller with improved architecture."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = DisplayConfig()
        self.stream_config = StreamConfig()
        self.buttons = [
            Button('Home', 3),
            Button('Back', 4),
            Button('Power', 26, color=(80, 50, 50))
        ]
        
        self.video_stream = VideoStream(self.config, self.stream_config)
        self.gesture_recognizer = GestureRecognizer(self.config)
        self.ui_renderer = UIRenderer(self.config, self.buttons)
        
        self.frame_queue = queue.Queue(maxsize=2)
        self.frame_reader: Optional[FrameReader] = None
        self.is_running = False
        
        # Performance monitoring
        self.fps_counter = {
            'start_time': time.time(),
            'frame_count': 0,
            'current_fps': 0.0
        }
        
        self.window_name = "Enhanced Android Watch Controller"
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                # Update configuration based on file
                logger.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
    
    def _check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        if not AdbInterface.check_device_connected():
            logger.error("No ADB device connected")
            return False
        
        logger.info("Prerequisites check passed")
        return True
    
    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter['frame_count'] += 1
        current_time = time.time()
        elapsed = current_time - self.fps_counter['start_time']
        
        if elapsed >= 1.0:
            self.fps_counter['current_fps'] = self.fps_counter['frame_count'] / elapsed
            self.fps_counter['frame_count'] = 0
            self.fps_counter['start_time'] = current_time
    
    def _handle_gesture(self, gesture_info: Dict[str, Any]):
        """Handle recognized gestures."""
        if gesture_info['type'] == 'button_gesture':
            button = self.buttons[gesture_info['button_index']]
            if gesture_info['gesture'] == GestureType.TAP:
                AdbInterface.key_event(button.keycode)
            elif gesture_info['gesture'] == GestureType.LONG_PRESS:
                logger.info(f"Long press on {button.name} button")
                # Handle special long press actions
                
        elif gesture_info['type'] == 'screen_gesture':
            gesture = gesture_info['gesture']
            if gesture == GestureType.TAP:
                x, y = gesture_info['position']
                AdbInterface.tap(x, y)
            elif gesture == GestureType.SWIPE:
                start_x, start_y = gesture_info['start']
                end_x, end_y = gesture_info['end']
                AdbInterface.swipe(start_x, start_y, end_x, end_y)
            elif gesture == GestureType.LONG_PRESS:
                x, y = gesture_info['position']
                AdbInterface.long_press(x, y)
            elif gesture == GestureType.SCROLL:
                start_x, start_y = gesture_info['start']
                end_x, end_y = gesture_info['end']
                AdbInterface.swipe(start_x, start_y, end_x, end_y)
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param):
        """Handle mouse events."""
        # Prevent OpenCV context menu on right-click
        if event in [cv2.EVENT_RBUTTONDOWN, cv2.EVENT_RBUTTONUP]:
            gesture_info = self.gesture_recognizer.process_mouse_event(event, x, y, self.buttons)
            if gesture_info:
                self._handle_gesture(gesture_info)
            return
        
        gesture_info = self.gesture_recognizer.process_mouse_event(event, x, y, self.buttons)
        if gesture_info:
            self._handle_gesture(gesture_info)
    
    def run(self):
        """Main application loop with enhanced stream management."""
        if not self._check_prerequisites():
            return
        
        logger.info("Starting Enhanced Android Watch Controller")
        
        # Setup OpenCV window
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        self.is_running = True
        last_frame = np.zeros((self.config.watch_height, self.config.watch_width, 3), dtype=np.uint8)
        retry_count = 0
        max_retries = 5
        last_stream_restart = 0
        stream_restart_cooldown = 5.0  # Wait 5 seconds between restart attempts
        

        try:
            while self.is_running:
                current_time = time.time()
                
                # Handle stream connection with improved logic
                if not self.video_stream.is_running or (self.frame_reader and not self.frame_reader.running):
                    # Check cooldown period
                    if current_time - last_stream_restart < stream_restart_cooldown:
                        time.sleep(0.1)
                        continue
                    
                    if retry_count >= max_retries:
                        logger.error("Max retries reached. Waiting before trying again...")
                        time.sleep(10)  # Wait longer before retrying
                        retry_count = 0
                        continue
                    
                    logger.info(f"Attempting to start stream (attempt {retry_count + 1}/{max_retries})")
                    
                    # Clean up existing stream
                    self._cleanup_stream()
                    
                    # Try to start new stream
                    if self.video_stream.start():
                        self.frame_reader = FrameReader(self.video_stream, self.frame_queue)
                        self.frame_reader.start()
                        retry_count = 0
                        last_stream_restart = current_time
                        logger.info("Stream restarted successfully")
                    else:
                        retry_count += 1
                        logger.warning(f"Stream start failed, retry in {stream_restart_cooldown}s")
                        last_stream_restart = current_time
                        continue
                
                # Get frame with timeout
                frame_start_time = time.time()
                try:
                    current_frame = self.frame_queue.get(timeout=0.1)
                    last_frame = current_frame
                    
                    # Record performance metrics
                    frame_process_time = time.time() - frame_start_time
                    
                except queue.Empty:
                    # Use last frame if no new frame available
                    current_frame = last_frame
                
                # Update performance metrics
                self._update_fps()
                
                # Get comprehensive statistics
                stats = None
                if self.frame_reader:
                    stats = self.frame_reader.get_stats()
                    
                    # Auto-restart stream if it's clearly broken
                    time_since_frame = stats.get('time_since_last_frame', 0)
                    if time_since_frame > 10.0:  # 10 seconds without frames
                        logger.warning("Stream appears frozen, forcing restart")
                        self._cleanup_stream()
                        continue
                
                # Render UI with enhanced information
                ui_frame = self.ui_renderer.render_frame(
                    current_frame, 
                    self.fps_counter['current_fps'],
                    stats
                )
                
                cv2.imshow(self.window_name, ui_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    print(convert_opencv_to_adb_keycode_string(key),key)
                    AdbInterface.key_event(convert_opencv_to_adb_keycode_string(key))
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self._cleanup()
    
    def _cleanup_stream(self):
        """Clean up current stream and frame reader."""
        if self.frame_reader and self.frame_reader.is_alive():
            self.frame_reader.stop()
            self.frame_reader.join(timeout=3)
        
        self.video_stream.stop()
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def _restart_stream(self):
        """Restart the video stream."""
        if self.frame_reader:
            self.frame_reader.stop()
            self.frame_reader.join()
        
        self.video_stream.stop()
        time.sleep(1)  # Give time for cleanup
    
    def _save_screenshot(self, frame: np.ndarray):
        """Save current frame as screenshot."""
        if frame is not None:
            timestamp = int(time.time())
            filename = f"watch_screenshot_{timestamp}.png"
            cv2.imwrite(filename, frame)
            logger.info(f"Screenshot saved as {filename}")
    
    def _cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        self.is_running = False
        
        if self.frame_reader and self.frame_reader.is_alive():
            self.frame_reader.stop()
            self.frame_reader.join(timeout=5)
        
        self.video_stream.stop()
        cv2.destroyAllWindows()
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Cleanup completed")


class ConfigManager:
    """Manages application configuration with validation."""
    
    DEFAULT_CONFIG = {
        "display": {
            "watch_width": 480,
            "watch_height": 480,
            "scale": 1.0,
            "button_width": 80,
            "button_height": 50,
            "button_spacing": 10
        },
        "stream": {
            "bitrate": "16m",
            "buffer_size": 10000000,
            "retry_attempts": 3,
            "reconnect_delay": 2.0
        },
        "gestures": {
            "long_press_duration": 0.6,
            "movement_threshold": 8,
            "swipe_min_distance": 15,
            "scroll_min_distance": 10
        },
        "buttons": [
            {"name": "Home", "keycode": 3, "color": [50, 50, 50]},
            {"name": "Back", "keycode": 4, "color": [50, 50, 50]},
            {"name": "Power", "keycode": 26, "color": [80, 50, 50]},
            {"name": "Menu", "keycode": 82, "color": [50, 80, 50]},
            {"name": "Volume+", "keycode": 24, "color": [50, 50, 80]},
            {"name": "Volume-", "keycode": 25, "color": [50, 50, 80]}
        ],
        "logging": {
            "level": "INFO",
            "file": "watch_controller.log",
            "max_size": 10485760,
            "backup_count": 3
        }
    }
    
    @classmethod
    def load_config(cls, config_path: str) -> dict:
        """Load configuration from file with validation."""
        config = cls.DEFAULT_CONFIG.copy()
        
        if Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                config = cls._merge_configs(config, user_config)
                logger.info(f"Configuration loaded from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info("Config file not found, using defaults")
            cls.save_config(config_path, config)
        
        return cls._validate_config(config)
    
    @classmethod
    def save_config(cls, config_path: str, config: dict):
        """Save configuration to file."""
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
    
    @classmethod
    def _merge_configs(cls, base: dict, overlay: dict) -> dict:
        """Recursively merge two configuration dictionaries."""
        result = base.copy()
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def _validate_config(cls, config: dict) -> dict:
        """Validate configuration values."""
        # Validate display settings
        display = config.get('display', {})
        display['watch_width'] = max(240, min(display.get('watch_width', 480), 1920))
        display['watch_height'] = max(240, min(display.get('watch_height', 480), 1920))
        display['scale'] = max(0.5, min(display.get('scale', 1.0), 3.0))
        
        # Validate gesture settings
        gestures = config.get('gestures', {})
        gestures['long_press_duration'] = max(0.1, min(gestures.get('long_press_duration', 0.6), 2.0))
        gestures['movement_threshold'] = max(2, min(gestures.get('movement_threshold', 8), 50))
        
        return config



def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Android Watch Controller")
    parser.add_argument('--config', '-c', default='watch_controller_config.json',
                       help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--check-device', action='store_true',
                       help='Check device connection and exit')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.check_device:
        if AdbInterface.check_device_connected():
            print("✓ ADB device is connected and ready")
            return 0
        else:
            print("✗ No ADB device found")
            print("Make sure your device is connected and USB debugging is enabled")
            return 1
    
    try:
        # Load configuration
        config = ConfigManager.load_config(args.config)
        
        # Create and run controller
        controller = WatchController(args.config)
        controller.run()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())