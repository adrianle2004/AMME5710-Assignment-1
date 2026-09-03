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

def show(axis, image: np.ndarray, title: str, gray: bool = False) -> None:
    """
    Draws an image onto a matplotlib axis with a title and no tick marks.

    OpenCV stores colour images as BGR whereas matplotlib expects RGB, so
    colour images are converted before being plotted.

    Args:
        axis: The matplotlib axis to draw on.
        image (np.ndarray): The image to display, BGR if colour.
        title (str): Title placed above the image.
        gray (bool): True if the image is single channel.
    """
    if gray:
        axis.imshow(image, cmap='gray')
    else:
        axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])


def morphological_operations(bin_board: np.ndarray, n: int) -> np.ndarray:
    """
    Applies morphological operations to the binary board image to clean up noise.

    Args:
        bin_board (np.ndarray): The binary image of the board.
        n (int): The number of times to apply the morphological operations.

    Returns:
        np.ndarray: The cleaned binary image after morphological operations.
    """
    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # Apply morphological operations
    for _ in range(n):
        bin_board = cv2.morphologyEx(bin_board, cv2.MORPH_OPEN, kernel)
        bin_board = cv2.morphologyEx(bin_board, cv2.MORPH_CLOSE, kernel)

    return bin_board


def find_board_corners(bin_board: np.ndarray) -> np.ndarray:
    """
    Finds the four corners of the game board in a binary board mask.

    The largest external contour is taken to be the board, and is then
    simplified with the Douglas-Peucker algorithm until only four vertices
    remain, which are the corners.

    Args:
        bin_board (np.ndarray): Binary image with the board pixels set to 255.

    Returns:
        np.ndarray: A 4-by-2 float32 array of (x, y) corner coordinates, ordered
            (upper left, upper right, lower left, lower right) to match the
            ordering used in "board_corners.pkl".
    """
    # The board is by far the largest blob, so any remaining specks of
    # background that survived the colour threshold are discarded here.
    contours, _ = cv2.findContours(bin_board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    board_contour = max(contours, key=cv2.contourArea)

    # cv2.approxPolyDP simplifies the outline by dropping vertices that sit
    # within "epsilon" pixels of the line between their neighbours. Epsilon is
    # set as a fraction of the contour perimeter so that it scales with the size
    # of the board in the image. The fraction is swept upwards until the outline
    # collapses to a quadrilateral, rather than trusting a single hard-coded
    # value to work on every image.
    approx = None
    for epsilon_fraction in np.arange(0.01, 0.10, 0.005):
        epsilon = epsilon_fraction * cv2.arcLength(board_contour, True)
        candidate = cv2.approxPolyDP(board_contour, epsilon, True)
        if len(candidate) == 4:
            approx = candidate
            break

    # If no epsilon gives a clean quad, fall back to the minimum-area enclosing
    # rectangle, which always has exactly four corners.
    if approx is None:
        approx = cv2.boxPoints(cv2.minAreaRect(board_contour))

    points = approx.reshape(4, 2).astype(np.float32)

    # approxPolyDP returns the corners in the order the contour was traced,
    # which depends on the board's rotation and on where the trace started. They
    # are sorted here so the output ordering is always the same: the upper left
    # corner has the smallest (x + y) and the lower right the largest, while the
    # upper right has the smallest (y - x) and the lower left the largest.
    coordinate_sum = points.sum(axis=1)
    coordinate_diff = points[:, 1] - points[:, 0]
    upper_left = points[np.argmin(coordinate_sum)]
    lower_right = points[np.argmax(coordinate_sum)]
    upper_right = points[np.argmin(coordinate_diff)]
    lower_left = points[np.argmax(coordinate_diff)]

    return np.array([upper_left, upper_right, lower_left, lower_right], np.float32)

def rectified_board(img: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Rectifies the board in the given image to a fronto-parallel view.

    Args:
        img (np.ndarray): The input BGR image containing the board.
        corners (np.ndarray): A 4-by-2 float32 array of (x, y) corner coordinates,
            ordered (upper left, upper right, lower left, lower right).

    Returns:
        tuple[np.ndarray, int, int]: The rectified board image and its dimensions.
    """
# ---------------------------------------------------------------------------
# Step 3 - rectify the board to a fronto-parallel view
# ---------------------------------------------------------------------------

    # Here we first find the size of the edges of the board in the image, then we use that to compute the perspective transform matrix. Finally, we apply the perspective transform to get a top-down view of the board.
    up_edge = np.linalg.norm(corners[1] - corners[0])
    down_edge = np.linalg.norm(corners[3] - corners[2])

    max_width = int(max(up_edge, down_edge))

    left_edge = np.linalg.norm(corners[2] - corners[0])
    right_edge = np.linalg.norm(corners[3] - corners[1])

    max_height = int(max(left_edge, right_edge))

    input_pts = corners
    output_pts = np.array([[0, 0], [max_width - 1, 0], [0, max_height - 1], [max_width - 1, max_height - 1]], dtype=np.float32)


    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts,output_pts)

    # Perspective transform the original image to get a top-down view of the board
    warped_board = cv2.warpPerspective(img, M, (max_width, max_height))

    return warped_board, max_width, max_height

def circle_encoder(img: np.ndarray) -> tuple:
    """
    Detects the 42 circular cells in a rectified board image with a Hough
    transform.

    The search radius is derived from the board geometry rather than hard coded:
    a rectified board is 7 cells wide and 6 tall, so the cell pitch is known from
    the image size, and a cell opening is a little over half a pitch across.

    Args:
        img (np.ndarray): The rectified (perspective corrected) BGR board image.

    Returns:
        tuple: (output_img, circles) where output_img has the detections drawn on
            it and circles is an N-by-3 array of (x, y, radius), or None if the
            transform found nothing.
    """
    # Work out how big a cell should be from the size of the rectified board
    cell_width = img.shape[1] / 7
    cell_height = img.shape[0] / 6
    cell_pitch = min(cell_width, cell_height)
    expected_radius = 0.30 * cell_pitch

    # Convert to grayscale and blur. The blur suppresses the wood grain visible
    # through the empty cells, which otherwise generates spurious edge votes.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 9)

    # Use HoughCircles to detect circles in the image. minDist is set just below
    # one cell pitch so that two circles cannot be reported inside one cell,
    # and the radius range brackets the expected radius.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=int(0.70 * cell_pitch),
                               param1=100, param2=40,
                               minRadius=int(0.75 * expected_radius),
                               maxRadius=int(1.35 * expected_radius))

    # Create a copy of the original image to draw on
    output_img = img.copy()

    # If some circles are detected, draw them on the output image. Each circle is
    # drawn at its own detected radius r, outlined rather than filled, so that
    # the cell contents stay visible underneath.
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output_img, (x, y), r, (0, 255, 0), 5)
            cv2.circle(output_img, (x, y), 4, (0, 0, 255), -1)

    return output_img, circles


def find_cells_by_mask(warped_board: np.ndarray) -> tuple:
    """
    Detects the 42 cells in a rectified board image by inverting the board
    colour mask.

    Every cell opening is a hole in the blue plastic, so whatever is *not*
    selected by the board colour threshold, but lies inside the board outline,
    is a cell. This reuses the colour selection already performed rather than
    searching for a shape, and unlike the Hough transform it does not care
    whether the openings are perfectly circular.

    Args:
        warped_board (np.ndarray): The rectified BGR board image.

    Returns:
        tuple: (output_img, centres) where centres is an N-by-2 array of cell
            centre coordinates.
    """
    cell_area = (warped_board.shape[1] / 7) * (warped_board.shape[0] / 6)

    # Anything that is not board-coloured is a cell opening
    board_mask = (histogram_color_select(warped_board).any(axis=2)).astype(np.uint8) * 255
    holes = cv2.bitwise_not(board_mask)

    # Opening removes the thin sliver of non-board pixels around the border and
    # any speckle, leaving 42 solid blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, kernel)

    # Keep only blobs large enough to be a real cell
    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 0.10 * cell_area]

    output_img = warped_board.copy()
    centres = []
    for contour in contours:
        moments = cv2.moments(contour)
        centre_x = int(moments['m10'] / moments['m00'])
        centre_y = int(moments['m01'] / moments['m00'])
        centres.append([centre_x, centre_y])
        cv2.drawContours(output_img, [contour], -1, (0, 255, 0), 5)
        cv2.circle(output_img, (centre_x, centre_y), 4, (0, 0, 255), -1)

    return output_img, np.array(centres)


img = cv2.imread('connect_four_images_A1/013.jpg')

# Step 0 - isolate the board by its colour
board = histogram_color_select(img)

# Step 1 - reduce to a binary mask and clean it up with morphology
bin_board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
bin_board = cv2.threshold(bin_board, 1, 255, cv2.THRESH_BINARY)[1]
bin_board = morphological_operations(bin_board, 5)

# Step 2 - trace the board outline and reduce it to four corners
contours, _ = cv2.findContours(bin_board, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
board_contour = max(contours, key=cv2.contourArea)
corners = find_board_corners(bin_board)

print('detected corners (UL, UR, LL, LR):')
print(corners)


# ---------------------------------------------------------------------------
# Figure 1 - the colour selection step
# ---------------------------------------------------------------------------
figure, axes = plt.subplots(1, 2, figsize=(9, 7), layout='constrained')
show(axes[0], img, 'Original')
show(axes[1], board, 'Board detection (colour selected)')
figure.suptitle('Step 0 - isolating the board by colour')


# ---------------------------------------------------------------------------
# Figure 2 - the binary mask, the traced contour, and the recovered corners
# ---------------------------------------------------------------------------

# Draw the traced outline on its own copy of the original
contour_overlay = img.copy()
cv2.drawContours(contour_overlay, [board_contour], -1, (0, 255, 0), 12)

# Draw the four corners on another copy. The corners are reordered to
# UL, UR, LR, LL so that polylines traces the outline of the board rather
# than crossing over itself in a bow-tie.
corner_overlay = img.copy()
cv2.polylines(corner_overlay, [corners[[0, 1, 3, 2]].astype(np.int32)], True, (0, 255, 0), 8)
for x, y in corners:
    cv2.circle(corner_overlay, (int(x), int(y)), 35, (0, 0, 255), -1)

figure, axes = plt.subplots(1, 3, figsize=(13, 6), layout='constrained')
show(axes[0], bin_board, 'Binary mask (after morphology)', gray=True)
show(axes[1], contour_overlay, 'Largest contour (%d points)' % len(board_contour))
show(axes[2], corner_overlay, 'Original with found corners')

# Label each corner on the third panel so the ordering can be checked by eye
for label, (x, y) in zip(['UL', 'UR', 'LL', 'LR'], corners):
    axes[2].annotate(label, (x, y), color='yellow', fontsize=12, weight='bold',
                     xytext=(12, 12), textcoords='offset points')

figure.suptitle('Steps 1 and 2 - binary mask, contour and corner detection')


warped_board, max_width, max_height = rectified_board(img, corners)
print('warped board size: %d x %d px' % (max_width, max_height))

# ---------------------------------------------------------------------------
# Figure 3 - the original next to the rectified board
# ---------------------------------------------------------------------------
figure, axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
show(axes[0], img, 'Original (%d x %d)' % (img.shape[1], img.shape[0]))
show(axes[1], warped_board, 'Perspective corrected (%d x %d)' % (max_width, max_height))
figure.suptitle('Step 3 - rectifying the board with a perspective transform')

plt.show()

# ---------------------------------------------------------------------------
# Figure 4 - locating the 42 cells, two ways
# ---------------------------------------------------------------------------
hough_img, hough_circles = circle_encoder(warped_board)
mask_img, mask_centres = find_cells_by_mask(warped_board)

n_hough = 0 if hough_circles is None else len(hough_circles)
print('cells found - Hough: %d/42, mask inversion: %d/42' % (n_hough, len(mask_centres)))

figure, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
show(axes[0], hough_img, 'HoughCircles (%d / 42 found)' % n_hough)
show(axes[1], mask_img, 'Mask inversion (%d / 42 found)' % len(mask_centres))
figure.suptitle('Step 4 - locating the 42 cells')

plt.show()