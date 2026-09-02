import os
import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt

# Dictionary of chessboard corner coordinates for each image
with open('assign1Q2_validationdata/board_corners.pkl', 'rb') as f:
    board_corners = pickle.load(f)

# Dictionary of chessboard groundtruth states for each image
with open('assign1Q2_validationdata/board_states.pkl', 'rb') as f:
    board_groundtruth = pickle.load(f)

def histogram_color_select(frame: np.ndarray) -> np.ndarray:
    """
    Selects the color of the board in the given frame using histogram-based color selection.

    Args:
        frame (np.ndarray): The input BGR image frame.

    Returns:
        np.ndarray: The input frame with everything outside the selected colour
            range masked out (still in BGR).
    """
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate the hue histogram. OpenCV packs hue into 0-179 so that it fits
    # in a uint8, whereas saturation and value use the full 0-255.
    hist_h = cv2.calcHist([hsv_frame], [0], None, [180], [0, 180])

    # Find the peak hue within the blue range. argmax over the slice is an index
    # into the slice, so the slice start is added back on to get the hue bin.
    blue_hue_range = (100, 140)  # Approximate hue range for blue color
    peak_hue = blue_hue_range[0] + int(np.argmax(hist_h[blue_hue_range[0]:blue_hue_range[1]]))
    hue_lo, hue_hi = max(peak_hue - 10, 0), min(peak_hue + 10, 179)

    # The wall behind the board shares the board's hue but is washed out, and it
    # covers far more of the image, so it wins any histogram vote taken over all
    # blue-hued pixels. Saturation is what actually separates the two (the board
    # sits near 220, the wall near 60), so Otsu's method is used to find the
    # split point between the two saturation populations and everything below it
    # is discarded before the peaks are measured.
    hue_mask = cv2.inRange(hsv_frame, np.array([hue_lo, 0, 0], np.uint8),
                           np.array([hue_hi, 255, 255], np.uint8))
    blue_saturations = hsv_frame[:, :, 1][hue_mask > 0].reshape(-1, 1)
    sat_split, _ = cv2.threshold(blue_saturations, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Measure the saturation and value peaks over the saturated blue pixels only
    board_mask = cv2.inRange(hsv_frame, np.array([hue_lo, int(sat_split), 0], np.uint8),
                             np.array([hue_hi, 255, 255], np.uint8))
    peak_saturation = int(np.argmax(cv2.calcHist([hsv_frame], [1], board_mask, [256], [0, 256])))
    peak_value = int(np.argmax(cv2.calcHist([hsv_frame], [2], board_mask, [256], [0, 256])))

    # Create a mask based on the identified peak values. The value window is kept
    # wide because brightness varies a lot across the board with shading, while
    # hue and saturation stay tight. Bounds are clipped to the valid channel
    # ranges so they cannot wrap around when cast to uint8.
    lower_bound = np.array([hue_lo,
                            max(peak_saturation - 60, 0),
                            max(peak_value - 100, 0)], np.uint8)
    upper_bound = np.array([hue_hi,
                            min(peak_saturation + 60, 255),
                            min(peak_value + 100, 255)], np.uint8)
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Apply the mask to the original frame to extract the selected color
    board = cv2.bitwise_and(frame, frame, mask=mask)

    return board

def show(window_name: str, image: np.ndarray, max_height: int = 800) -> None:
    """
    Displays an image in a window, scaled down to fit on screen.

    The dataset images are 3472-by-2598, which is far taller than any monitor,
    so they are shrunk to at most max_height pixels tall before being shown.

    Args:
        window_name (str): Title of the display window.
        image (np.ndarray): The BGR image to display.
        max_height (int): Maximum height of the displayed window in pixels.
    """
    scale = min(1.0, max_height / image.shape[0])
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(window_name, image)


img = cv2.imread('connect_four_images_A1/013.jpg')

board = histogram_color_select(img)

show('Selected Board Color', board)
cv2.waitKey(0)
cv2.destroyAllWindows()
