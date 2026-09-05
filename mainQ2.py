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

def board_color_bounds(frame: np.ndarray) -> tuple:
    """
    Works out the HSV window that isolates the board in a single frame.

    This is the measurement half of the colour selection, split out from the
    masking so that the same numbers can be plotted as histograms without being
    recomputed by a second, possibly divergent, copy of the logic.

    Args:
        frame (np.ndarray): The input BGR image frame.

    Returns:
        tuple: (lower_bound, upper_bound, stats) where the bounds are uint8 HSV
            triples suitable for cv2.inRange, and stats is a dict holding the
            histograms and the thresholds derived from them, for plotting.
    """
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate the hue histogram. OpenCV packs hue into 0-179 so that it fits
    # in a uint8, whereas saturation and value use the full 0-255.
    hist_hue = cv2.calcHist([hsv_frame], [0], None, [180], [0, 180]).ravel()

    # Find the peak hue within the blue range. argmax over the slice is an index
    # into the slice, so the slice start is added back on to get the hue bin.
    # The lower edge sits at 90 rather than 100 because the board's blue actually
    # peaks at hue 94-98 across the fifteen images. A range starting at 100
    # clipped that peak and reported the boundary bin itself as the maximum on
    # every image, which happened to still work only because the +/-10 window
    # placed around it reached back far enough to cover the real peak.
    blue_hue_range = (90, 140)  # Approximate hue range for blue color
    peak_hue = blue_hue_range[0] + int(np.argmax(hist_hue[blue_hue_range[0]:blue_hue_range[1]]))
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
    hist_saturation = cv2.calcHist([hsv_frame], [1], board_mask, [256], [0, 256]).ravel()
    hist_value = cv2.calcHist([hsv_frame], [2], board_mask, [256], [0, 256]).ravel()
    peak_saturation = int(np.argmax(hist_saturation))
    peak_value = int(np.argmax(hist_value))

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

    # The saturation of every blue-hued pixel, before the Otsu split throws the
    # wall away. Kept only so the plot can show the two populations side by side.
    hist_blue_saturation = cv2.calcHist([hsv_frame], [1], hue_mask, [256], [0, 256]).ravel()

    stats = {'hsv_frame': hsv_frame,
             'hist_hue': hist_hue,
             'blue_hue_range': blue_hue_range,
             'peak_hue': peak_hue,
             'hue_lo': hue_lo,
             'hue_hi': hue_hi,
             'hist_blue_saturation': hist_blue_saturation,
             'hist_saturation': hist_saturation,
             'hist_value': hist_value,
             'sat_split': int(sat_split),
             'peak_saturation': peak_saturation,
             'peak_value': peak_value}

    return lower_bound, upper_bound, stats


def histogram_color_select(frame: np.ndarray) -> np.ndarray:
    """
    Selects the color of the board in the given frame using histogram-based color selection.

    Args:
        frame (np.ndarray): The input BGR image frame.

    Returns:
        np.ndarray: The input frame with everything outside the selected colour
            range masked out (still in BGR).
    """
    lower_bound, upper_bound, stats = board_color_bounds(frame)
    mask = cv2.inRange(stats['hsv_frame'], lower_bound, upper_bound)

    # Apply the mask to the original frame to extract the selected color
    board = cv2.bitwise_and(frame, frame, mask=mask)

    return board


def plot_color_histograms(axes: dict, frame: np.ndarray) -> None:
    """
    Draws the three histograms that the colour selection is built from, with the
    thresholds it derived marked on them.

    The hue panel shows where the blue peak was found; the saturation panel shows
    the two populations of blue-hued pixels, the wall and the board, and the Otsu
    split that separates them; the value panel shows the brightness spread of the
    board itself. On each, the shaded band is the window that ends up being
    passed to cv2.inRange.

    Args:
        axes (dict): Matplotlib axes keyed 'hue', 'saturation' and 'value'.
        frame (np.ndarray): The input BGR image frame.
    """
    lower_bound, upper_bound, stats = board_color_bounds(frame)

    # --- hue -------------------------------------------------------------
    # Plotted on a log scale because the background dominates the count by
    # orders of magnitude and would otherwise flatten the board's peak to
    # nothing.
    axis = axes['hue']
    axis.bar(np.arange(180), stats['hist_hue'], width=1.0, color='0.65')
    axis.axvspan(stats['blue_hue_range'][0], stats['blue_hue_range'][1],
                 color='tab:blue', alpha=0.10, label='blue search range')
    axis.axvspan(stats['hue_lo'], stats['hue_hi'], color='tab:blue', alpha=0.30,
                 label='accepted window %d-%d' % (stats['hue_lo'], stats['hue_hi']))
    axis.axvline(stats['peak_hue'], color='tab:red', lw=1.5,
                 label='peak hue = %d' % stats['peak_hue'])
    axis.set_yscale('log')
    axis.set_xlim(0, 179)
    axis.set_xlabel('hue (OpenCV 0-179)')
    axis.set_ylabel('pixels (log)')
    axis.set_title('Hue histogram (whole image)')
    axis.legend(fontsize=7, loc='upper right')

    # --- saturation ------------------------------------------------------
    # The grey histogram is every blue-hued pixel and is clearly bimodal: the
    # low mode is the wall, the high mode the board. The blue histogram is what
    # survives the Otsu split, and it is only over those pixels that the peak is
    # measured.
    axis = axes['saturation']
    axis.bar(np.arange(256), stats['hist_blue_saturation'], width=1.0,
             color='0.65', label='all blue-hued pixels')
    axis.bar(np.arange(256), stats['hist_saturation'], width=1.0,
             color='tab:blue', label='kept after Otsu split')
    axis.axvline(stats['sat_split'], color='tab:orange', lw=1.5,
                 label='Otsu split = %d' % stats['sat_split'])
    axis.axvspan(lower_bound[1], upper_bound[1], color='tab:blue', alpha=0.20,
                 label='accepted window %d-%d' % (lower_bound[1], upper_bound[1]))
    axis.axvline(stats['peak_saturation'], color='tab:red', lw=1.5,
                 label='peak = %d' % stats['peak_saturation'])
    axis.set_yscale('log')
    axis.set_xlim(0, 255)
    axis.set_xlabel('saturation')
    axis.set_ylabel('pixels (log)')
    axis.set_title('Saturation of blue-hued pixels')
    axis.legend(fontsize=7, loc='upper right')

    # --- value -----------------------------------------------------------
    # Measured over the board pixels only. The window is deliberately wide
    # because shading across the board moves brightness around a lot.
    axis = axes['value']
    axis.bar(np.arange(256), stats['hist_value'], width=1.0, color='tab:blue')
    axis.axvspan(lower_bound[2], upper_bound[2], color='tab:blue', alpha=0.20,
                 label='accepted window %d-%d' % (lower_bound[2], upper_bound[2]))
    axis.axvline(stats['peak_value'], color='tab:red', lw=1.5,
                 label='peak = %d' % stats['peak_value'])
    axis.set_xlim(0, 255)
    axis.set_xlabel('value')
    axis.set_ylabel('pixels')
    axis.set_title('Value over the board pixels')
    axis.legend(fontsize=7, loc='upper right')


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



def assign_cells_to_grid(centres: np.ndarray, board_shape: tuple) -> np.ndarray:
    """
    Assigns detected cell centres to their (row, column) position in the 6-by-7
    playing grid.

    Once the board has been rectified it is a plain axis-aligned rectangle that
    is exactly 7 cells wide and 6 cells tall, so a centre's position in the grid
    follows directly from where it falls in the image: dividing its x coordinate
    by the cell width gives the column and its y coordinate by the cell height
    gives the row. No sorting or clustering of the detections is needed, and a
    missing detection simply leaves its slot empty instead of shifting every
    later cell along by one, which is what a sort-based assignment would do.

    Args:
        centres (np.ndarray): An N-by-2 array of (x, y) cell centre coordinates
            in the rectified board image.
        board_shape (tuple): The shape of the rectified board image, i.e.
            (height, width) or (height, width, channels).

    Returns:
        np.ndarray: A 6-by-7 int array holding, for each grid position, the
            index into "centres" of the cell that was found there, or -1 if no
            cell was detected at that position. Row 0 is the top of the board
            and column 0 is the left, matching "board_states.pkl".
    """
    height, width = board_shape[:2]
    cell_width = width / 7
    cell_height = height / 6

    grid = np.full((6, 7), -1, np.int32)
    for index, (x, y) in enumerate(centres):
        # Clipping guards against a centre sitting a pixel outside the board,
        # which would otherwise index off the end of the grid.
        column = int(np.clip(x // cell_width, 0, 6))
        row = int(np.clip(y // cell_height, 0, 5))
        grid[row, column] = index

    return grid


def classify_cell_colour(hsv_board: np.ndarray, centre: np.ndarray,
                         sample_radius: int) -> int:
    """
    Decides whether a single cell is empty, yellow or red.

    A disc of pixels is taken from the middle of the cell and reduced to a
    single representative colour with the median, which ignores the specular
    highlight on the token and the shadow around the rim of the opening in a way
    that the mean would not. The decision is then made in HSV: saturation says
    whether a token is present at all, because both token colours are strongly
    saturated while the background seen through an empty opening is washed out,
    and hue says which token it is. Hue is measured as a distance around the
    colour circle rather than a plain range, because red straddles the 0/180
    wrap-around point in OpenCV's packing and so appears at both ends.

    The thresholds below were set from the measured spread over all 630 cells of
    the validation set:

        empty   hue 12-99, hue distance to red 12-89, saturation   8-177
        yellow  hue 25-31, hue distance to red 25-31, saturation 206-238
        red     hue  0- 2 or 178-179, distance 0-2,   saturation 172-238

    Both gates are needed. Saturation alone cannot separate empty from red,
    because the wooden table seen through an opening reaches saturation 177
    while the darkest red token sits at 172, but their hues are 12 apart at the
    closest. Hue alone cannot separate empty from yellow, because 38 of the 358
    empty cells have a hue inside the yellow band, but every one of them is
    washed out to below saturation 150 while no yellow token falls below 206.
    Using both leaves a wide margin on either decision.

    Args:
        hsv_board (np.ndarray): The rectified board image converted to HSV.
        centre (np.ndarray): The (x, y) centre of the cell to classify.
        sample_radius (int): Radius in pixels of the disc to sample.

    Returns:
        int: 0 if the cell is empty, 1 if it holds a yellow token, 2 if red.
    """
    # Sample a disc from the middle of the cell and take the median colour
    sample_mask = np.zeros(hsv_board.shape[:2], np.uint8)
    cv2.circle(sample_mask, (int(centre[0]), int(centre[1])), sample_radius, 255, -1)
    hue, saturation, _ = np.median(hsv_board[sample_mask > 0], axis=0)

    # A washed out cell has nothing in it
    if saturation < 150:
        return 0

    # Red wraps around the end of the hue axis, so measure how far the hue is
    # from 0 going either way around the circle
    hue_distance_to_red = min(hue, 180 - hue)
    if hue_distance_to_red <= 6:
        return 2
    if 20 <= hue <= 40:
        return 1

    # Saturated but neither token colour, so treat it as empty
    return 0


def encode_board_state(warped_board: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """
    Builds the 6-by-7 board state from the detected cell centres.

    Args:
        warped_board (np.ndarray): The rectified BGR board image.
        centres (np.ndarray): An N-by-2 array of detected cell centres.

    Returns:
        np.ndarray: A 6-by-7 int array of cell states, 0 empty, 1 yellow, 2 red.
    """
    grid = assign_cells_to_grid(centres, warped_board.shape)
    hsv_board = cv2.cvtColor(warped_board, cv2.COLOR_BGR2HSV)

    # Sample well inside the opening so the rim of the board is never included
    cell_pitch = min(warped_board.shape[1] / 7, warped_board.shape[0] / 6)
    sample_radius = int(0.20 * cell_pitch)

    board_state = np.zeros((6, 7), np.int32)
    for row in range(6):
        for column in range(7):
            index = grid[row, column]
            # A grid position with no detection is left as empty, which is the
            # correct guess: a cell holding a token is a solid disc of strong
            # colour and is the easiest kind of cell to detect.
            if index >= 0:
                board_state[row, column] = classify_cell_colour(
                    hsv_board, centres[index], sample_radius)

    return board_state


def detect_board(img: np.ndarray) -> tuple:
    """
    Runs the geometric half of the pipeline on a photograph: isolate the board by
    colour, clean the mask up with morphology, trace its outline and reduce it to
    four corners, rectify it to a fronto-parallel view, and locate the 42
    openings within it.

    It is split out from board_state_from_image so that the accuracy measurement
    can reach the intermediate corners without running a second copy of the
    pipeline that could drift out of step with this one.

    Args:
        img (np.ndarray): A colour (BGR) photograph containing the board.

    Returns:
        tuple: (warped_board, corners, centres) - the rectified board image, the
            four detected corners ordered (UL, UR, LL, LR), and an N-by-2 array
            of cell centres in the rectified image.
    """
    # Step 0 - isolate the board by its colour
    board = histogram_color_select(img)

    # Step 1 - reduce to a binary mask and clean it up with morphology
    bin_board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    bin_board = cv2.threshold(bin_board, 1, 255, cv2.THRESH_BINARY)[1]
    bin_board = morphological_operations(bin_board, 5)

    # Step 2 and 3 - find the corners and rectify
    corners = find_board_corners(bin_board)
    warped_board, _, _ = rectified_board(img, corners)

    # Step 4 - find the cells
    _, centres = find_cells_by_mask(warped_board)

    return warped_board, corners, centres


def board_state_from_image(img: np.ndarray) -> np.ndarray:
    """
    Determines the state of the connect four board in a photograph.

    This is the function the brief asks for: it runs the whole pipeline, from the
    raw photograph through to the encoded board, using no validation data.

    Args:
        img (np.ndarray): A colour (BGR) photograph containing the board.

    Returns:
        np.ndarray: A 6-by-7 int array of cell states, 0 empty, 1 yellow, 2 red,
            with row 0 the top of the board and column 0 the left.
    """
    warped_board, _, centres = detect_board(img)

    # Step 5 and 6 - place the cells on the grid and read their colours
    return encode_board_state(warped_board, centres)


def board_detection_accuracy(img: np.ndarray, true_state: np.ndarray,
                             true_corners: np.ndarray = None) -> dict:
    """
    Measures how well the pipeline recovered the board in a single image.

    The headline number is the Board Accuracy defined in the brief: the
    percentage of the 42 cells whose state was identified correctly. Two
    diagnostics are reported alongside it, because when a board comes out wrong
    it is usually one of these that explains why rather than the colour
    classification itself:

    - "cells_detected", the number of openings Step 4 found. Anything short of 42
      means a cell was never examined at all and was left at its default of
      empty.
    - "corner_error", the mean distance in pixels between the four detected
      corners and the hand-labelled ones, if those are supplied. A large value
      means the rectification was fed a bad quadrilateral, which shifts every
      cell centre and can misalign the whole grid.

    Args:
        img (np.ndarray): A colour (BGR) photograph containing the board.
        true_state (np.ndarray): The ground-truth 6-by-7 state for that image.
        true_corners (np.ndarray): Optionally, the ground-truth 4-by-2 corners,
            ordered (UL, UR, LL, LR), to measure the corner error against.

    Returns:
        dict: The estimated state, the number of cells correct, the board
            accuracy as a percentage, whether the board was perfect, the number
            of cells detected, and the corner error if it could be measured.
    """
    warped_board, corners, centres = detect_board(img)
    estimated_state = encode_board_state(warped_board, centres)

    true_state = np.asarray(true_state)
    correct_cells = int((estimated_state == true_state).sum())

    result = {'estimated_state': estimated_state,
              'correct_cells': correct_cells,
              'total_cells': int(true_state.size),
              'board_accuracy': 100.0 * correct_cells / true_state.size,
              'perfect': correct_cells == true_state.size,
              'cells_detected': len(centres),
              'corner_error': None}

    # Mean Euclidean distance between corresponding corners. Both are in the
    # same (UL, UR, LL, LR) order, so they can be subtracted row by row.
    if true_corners is not None:
        true_corners = np.asarray(true_corners, np.float32)
        result['corner_error'] = float(
            np.mean(np.linalg.norm(corners - true_corners, axis=1)))

    return result


if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # Accuracy over the validation set
    # -----------------------------------------------------------------------
    image_names = sorted(board_groundtruth)
    board_accuracies = []
    perfect_boards = 0

    print('%-10s %-9s %-9s %s' % ('image', 'cells', 'corner px', 'board accuracy'))
    for name in image_names:
        img = cv2.imread(os.path.join('connect_four_images_A1', name))
        result = board_detection_accuracy(img, board_groundtruth[name],
                                          board_corners[name])

        board_accuracies.append(result['board_accuracy'])
        perfect_boards += result['perfect']

        print('%-10s %2d/42     %6.1f    %6.2f %%   (%d/%d cells)'
              % (name, result['cells_detected'], result['corner_error'],
                 result['board_accuracy'], result['correct_cells'],
                 result['total_cells']))

    # Average Board Accuracy is the mean of the per image accuracies, and
    # Overall Accuracy is the percentage of images recovered without a single
    # cell wrong.
    print()
    print('Average Board Accuracy: %.2f %%' % np.mean(board_accuracies))
    print('Overall Accuracy:       %.2f %% (%d/%d boards fully correct)'
          % (100.0 * perfect_boards / len(image_names), perfect_boards, len(image_names)))

    # -----------------------------------------------------------------------
    # Walk one image through the pipeline for the figures
    # -----------------------------------------------------------------------
    demo_name = '013.jpg'
    img = cv2.imread(os.path.join('connect_four_images_A1', demo_name))

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

    # -----------------------------------------------------------------------
    # Figure 1 - the colour selection step
    # -----------------------------------------------------------------------
    # The images go on the top row and the histograms they were derived from on
    # the bottom, so the thresholds can be read against the result they produce.
    mosaic = [['original', 'original', 'selected', 'selected'],
              ['hue', 'hue', 'saturation', 'value']]
    figure, axes = plt.subplot_mosaic(mosaic, figsize=(15, 9), layout='constrained')
    show(axes['original'], img, 'Original')
    show(axes['selected'], board, 'Board detection (colour selected)')
    plot_color_histograms(axes, img)
    figure.suptitle('Step 0 - isolating the board by colour')

    # -----------------------------------------------------------------------
    # Figure 2 - the binary mask, the traced contour, and the recovered corners
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Figure 3 - the original next to the rectified board
    # -----------------------------------------------------------------------
    warped_board, max_width, max_height = rectified_board(img, corners)

    figure, axes = plt.subplots(1, 2, figsize=(12, 6), layout='constrained')
    show(axes[0], img, 'Original (%d x %d)' % (img.shape[1], img.shape[0]))
    show(axes[1], warped_board, 'Perspective corrected (%d x %d)' % (max_width, max_height))
    figure.suptitle('Step 3 - rectifying the board with a perspective transform')

    # -----------------------------------------------------------------------
    # Figure 4 - locating the 42 cells, two ways
    # -----------------------------------------------------------------------
    hough_img, hough_circles = circle_encoder(warped_board)
    mask_img, mask_centres = find_cells_by_mask(warped_board)
    n_hough = 0 if hough_circles is None else len(hough_circles)

    figure, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    show(axes[0], hough_img, 'HoughCircles (%d / 42 found)' % n_hough)
    show(axes[1], mask_img, 'Mask inversion (%d / 42 found)' % len(mask_centres))
    figure.suptitle('Step 4 - locating the 42 cells')

    # -----------------------------------------------------------------------
    # Figure 5 - the grid assignment and the colour read off each cell
    # -----------------------------------------------------------------------
    grid = assign_cells_to_grid(mask_centres, warped_board.shape)
    estimated_state = encode_board_state(warped_board, mask_centres)
    true_state = np.array(board_groundtruth[demo_name])

    # Draw the cell boundaries and label every opening with its grid position
    grid_overlay = warped_board.copy()
    cell_width = warped_board.shape[1] / 7
    cell_height = warped_board.shape[0] / 6
    for column in range(1, 7):
        x = int(column * cell_width)
        cv2.line(grid_overlay, (x, 0), (x, warped_board.shape[0]), (0, 255, 255), 4)
    for row in range(1, 6):
        y = int(row * cell_height)
        cv2.line(grid_overlay, (0, y), (warped_board.shape[1], y), (0, 255, 255), 4)
    for row in range(6):
        for column in range(7):
            index = grid[row, column]
            if index >= 0:
                x, y = mask_centres[index]
                cv2.circle(grid_overlay, (int(x), int(y)), 8, (0, 0, 255), -1)
                cv2.putText(grid_overlay, '%d,%d' % (row, column), (int(x) - 55, int(y) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)

    # Draw the decoded state as a schematic board
    swatches = {0: (0.90, 0.90, 0.90), 1: (0.95, 0.80, 0.10), 2: (0.85, 0.15, 0.15)}
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    show(axes[0], grid_overlay, 'Cells assigned to the 6 x 7 grid')

    axes[1].set_xlim(-0.5, 6.5)
    axes[1].set_ylim(5.5, -0.5)
    axes[1].set_aspect('equal')
    axes[1].set_xticks(range(7))
    axes[1].set_yticks(range(6))
    for row in range(6):
        for column in range(7):
            value = int(estimated_state[row, column])
            axes[1].add_patch(plt.Circle((column, row), 0.42, color=swatches[value],
                                         ec='0.3'))
            # Ring any cell that disagrees with the ground truth
            if value != true_state[row, column]:
                axes[1].add_patch(plt.Circle((column, row), 0.47, fill=False,
                                             ec='magenta', lw=3))
    correct_cells = int((estimated_state == true_state).sum())
    axes[1].set_title('Decoded board state (%d/42 cells correct)' % correct_cells)
    figure.suptitle('Steps 5 and 6 - grid assignment and colour encoding for %s' % demo_name)

    plt.show()
