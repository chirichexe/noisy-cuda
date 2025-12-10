import subprocess
import time
import os
import sys
import cv2  
from PIL import Image
import numpy as np
import argparse # Needed for command-line argument parsing

# --- CONFIGURATION ---

# Base path structure for the executable
BASE_EXECUTABLE_PATH = "../../build/{impl}/noisy_cuda"

# Default runtime implementation/backend
DEFAULT_IMPL = "cpp"
# Default noise seed
DEFAULT_SEED = 1234

# Base parameters
OUTPUT_FILENAME = "temp_noise_frame.png"
IMAGE_SIZE = "1024x1024"

# Perlin noise parameters
FREQUENCY = 1.0
AMPLITUDE = 1.0
OCTAVES = 1
PERSISTENCE = 0.5

# Animation variables (global state)
x_offset = 0  
y_offset = 0  
OFFSET_INCREMENT = 64

# --- FUNCTIONS ---

def generate_frame(x_offset, y_offset, impl_type, seed):
    """
    Executes the noisy_cuda binary with the specified parameters and saves the image.

    :param x_offset: The horizontal offset for noise generation.
    :param y_offset: The vertical offset for noise generation.
    :param impl_type: The implementation backend ("simd", "cpp", "cuda").
    :param seed: The integer seed for noise generation.
    """
    # Dynamic executable path based on implementation type
    executable_path = BASE_EXECUTABLE_PATH.format(impl=impl_type)
    
    # Format the offset as a string of INTEGERS.
    offset_str = f"{int(x_offset)},{int(y_offset)}" 
    
    command = [
        executable_path,
        str(seed), # The seed argument
        "--output", OUTPUT_FILENAME,
        "--size", IMAGE_SIZE,
        "--format", "png",
        "--frequency", str(FREQUENCY),
        "--amplitude", str(AMPLITUDE),
        "--persistence", str(PERSISTENCE),
        "--octaves", str(OCTAVES),
        "--offset", offset_str
    ]
    
    try:
        # Execute the command and check for errors
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        print(f"ERROR: Executable not found at '{executable_path}'", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR during noisy_cuda execution: {e}", file=sys.stderr)
        print(f"STDERR Output: {e.stderr}", file=sys.stderr)
        raise e 

def main_loop(impl_type, seed):
    """
    Main animation loop with OpenCV visualization and WASD controls.
    
    :param impl_type: The implementation backend to use.
    :param seed: The seed for the noise generator.
    """
    global x_offset, y_offset
    
    WINDOW_TITLE = f"Perlin Noise Animation | Impl: {impl_type} | Seed: {seed}"
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    
    print(f"--- Animation Started (OpenCV Visualization) ---")
    print(f"Controls: WASD to move offset, Q to quit.")
    print(f"Backend: {impl_type} | Seed: {seed}")

    try:
        while True:
            frame_start_time = time.time()
            
            # 1. Generate the frame
            # The executable path is resolved inside generate_frame using impl_type
            generate_frame(x_offset, y_offset, impl_type, seed)
            
            # 2. Load and render the image using OpenCV
            try:
                # Load the PNG image.
                img_pil = Image.open(OUTPUT_FILENAME)
                img_np = np.array(img_pil)
                
                # Convert the image format for cv2.imshow
                if img_np.ndim == 3:
                     # Assumes RGB (PIL) -> BGR (OpenCV)
                     frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                elif img_np.ndim == 2:
                     # Single channel (grayscale) -> BGR for display
                     frame = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                else:
                     raise ValueError("Unsupported image dimensions.")

                # Optional: Add offset and FPS text to the frame
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                text_offset = f"Offset X: {int(x_offset)}, Y: {int(y_offset)}"
                text_fps = f"FPS: {current_fps:.2f}"
                
                cv2.putText(frame, text_offset, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, text_fps, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display the image
                cv2.imshow(WINDOW_TITLE, frame)
                
                # 3. Handle keyboard input (WASD for movement, Q for quit)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break 
                elif key == ord('a'): # Left
                    x_offset -= OFFSET_INCREMENT
                elif key == ord('d'): # Right
                    x_offset += OFFSET_INCREMENT
                elif key == ord('w'): # Up
                    y_offset -= OFFSET_INCREMENT
                elif key == ord('s'): # Down
                    y_offset += OFFSET_INCREMENT

            except Exception as e:
                # Catch image loading/display errors
                print(f"ERROR during CV2 visualization or image loading: {e}", file=sys.stderr)
                time.sleep(0.1)
                continue
                
            # 4. Update FPS calculation
            frame_end_time = time.time()
            frame_time = frame_end_time - frame_start_time
            
            frame_count += 1
            
            if elapsed_time >= 1.0: 
                print(f"FPS: {current_fps:.2f} | Frame Time: {frame_time*1000:.2f} ms | Offset: ({int(x_offset)}, {int(y_offset)})") 
                frame_count = 0
                start_time = frame_end_time

    except KeyboardInterrupt:
        print("\n--- Animation terminated by user ---")
    except Exception as e:
        print(f"\nCRITICAL ERROR IN MAIN LOOP: {e}", file=sys.stderr)
        
    finally:
        # Clean up and close all OpenCV windows
        cv2.destroyAllWindows()
        # Clean up the temporary file
        if os.path.exists(OUTPUT_FILENAME):
            os.remove(OUTPUT_FILENAME)
        print("Exiting application.")

def check_executable_path(impl_type):
    """
    Verifies if the specified executable path exists.
    """
    executable_path = BASE_EXECUTABLE_PATH.format(impl=impl_type)
    if not os.path.exists(executable_path):
        print(f"ERROR: Executable verification failed. Path not found: '{executable_path}'.", file=sys.stderr)
        print("Please ensure the project has been built for the specified implementation ('cpp', 'simd', or 'cuda').", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(description="Real-time Perlin Noise visualization using an external backend.")
    parser.add_argument(
        '--impl', 
        type=str, 
        default=DEFAULT_IMPL, 
        choices=['simd', 'cpp', 'cuda'],
        help='The noise generation implementation backend (simd, cpp, cuda).'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=DEFAULT_SEED, 
        help='The integer seed for the noise generator.'
    )
    
    args = parser.parse_args()
    
    # 0. Initial checks and setup
    check_executable_path(args.impl)
    
    # 1. Run the main loop with parsed arguments
    main_loop(args.impl, args.seed)