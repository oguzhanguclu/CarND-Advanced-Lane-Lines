import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

nx = 9          # number of horizontal corners on the chessboard
ny = 6          # number of vertical corners on the chessboard

# each object point for calibration i.e. (0,0,0), (1,0,0), (2,0,0) ....,(7,6,0)
obj_point = np.zeros((ny*nx, 3), np.float32)
obj_point[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# lists for object and image points
object_points = []
image_points = []

# scaling factors for pixel to real world distance transformation
ym_per_pix = 30/720     # meters per pixel in y dimension
xm_per_pix = 3.7/700    # meters per pixel in x dimension

# input and output folders for camera calibration
calib_input_folder = 'camera_cal'
calib_output_folder = 'output_images/calibration'

print('Camera calibration...')

# load chessboard images in the calibration folder
for frame_name in os.listdir(calib_input_folder):
    if frame_name.endswith(".jpg"):
        image = cv2.imread(os.path.join(calib_input_folder, frame_name))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray_image, (nx, ny), None)

        # add object and image points
        if ret:
            print('Chessboard corners are found for', frame_name)
            object_points.append(obj_point)
            image_points.append(corners)

            # draw the corners and save image to the output folder
            cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            cv2.imwrite(os.path.join(calib_output_folder, frame_name[:-4] + '_corners.jpg'), image)
        else:
            print('Chessboard corners are not found for', frame_name)


# an example image for testing
test_image = cv2.imread('test_images/test5.jpg')
image_size = (test_image.shape[1], test_image.shape[0])

# perform camera calibration using image points found from chessboard images and the object points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

if ret:
    print('Camera is calibrated')
else:
    print('Camera can not be calibrated')


# function for thresholding the image given as parameter
def threshold_image(image, frame_id, s_thresh=(80, 255), r_thresh=(80, 255), sx_thresh=(10, 255), sy_thresh=(60, 255)):
    # get R channel in RGB color space
    r_channel = image[:, :, 2]

    # convert image to HLS color space and get the S and L channels
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s_channel = hls_image[:, :, 2]
    l_channel = hls_image[:, :, 1]

    # apply Sobel to L channel in x direction
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # threshold x gradient
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

    # apply Sobel to L channel in y direction
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)
    abs_sobely = np.absolute(sobely)
    scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))

    # threshold y gradient
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= sy_thresh[0]) & (scaled_sobely <= sy_thresh[1])] = 1

    # threshold S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # threshold R channel
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 1

    # combine thresholds in a binary image (Sobel x-y thresholds and color (S-R) thresholds are combined separately)
    combined_binary_image = np.zeros_like(sxbinary)
    combined_binary_image[((sybinary == 1) & (sxbinary == 1)) | ((r_binary == 1) & (s_binary == 1))] = 1

    '''# save original image and the thresholding result to the output folder
    print('Image is thresholded:', frame_id)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(combined_binary_image, cmap='gray')
    ax2.set_title('Thresholded Image', fontsize=20)
    fig1 = plt.gcf()
    plt.show(block=True)
    plt.draw()
    fig1.savefig(os.path.join(thresh_output_folder, str(frame_id) + '_thresholded.jpg'), dpi=100, 
                 bbox_inches='tight', pad_inches=0.3)'''

    return combined_binary_image


# applies perspective transform to the input image and returns the warped image
def warp_image(line_image, frame_id):
    # coordinates of the source points in the original image
    bottom_left_src = (190, line_image.shape[0])
    top_left_src = (585, 455)
    bottom_right_src = (1115, line_image.shape[0])
    top_right_src = (696, 455)

    # coordinates of the corresponding target points
    bottom_left_dst = (330, line_image.shape[0])
    top_left_dst = (330, 0)
    bottom_right_dst = (970, line_image.shape[0])
    top_right_dst = (970, 0)

    # create arrays for the source and target points
    src = np.float32([bottom_left_src, top_left_src, bottom_right_src, top_right_src])
    dst = np.float32([bottom_left_dst, top_left_dst, bottom_right_dst, top_right_dst])

    # compute the transform
    transform = cv2.getPerspectiveTransform(src, dst)

    # warp the image to get bird's eye view of the road
    img_size = line_image.shape[1], line_image.shape[0]
    warped_image = cv2.warpPerspective(line_image, transform, img_size, flags=cv2.INTER_LINEAR)

    # compute the inverse transform for projecting detected lines onto the original image
    inverse_transform = cv2.getPerspectiveTransform(dst, src)

    # draw lines on the input image
    #line_image = cv2.line(line_image, bottom_left_src, top_left_src, (0, 255, 0), 2)
    #line_image = cv2.line(line_image, bottom_right_src, top_right_src, (0, 255, 0), 2)

    '''# save original line image and the warped result to the output folder
    print('Image is warped:', frame_id)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    #ax1.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
    ax1.imshow(line_image, cmap='gray')
    ax1.set_title('Undistorted and Thresholded Image', fontsize=20)
    ax2.imshow(warped_image, cmap='gray')
    #ax2.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    ax2.set_title('Warped Image', fontsize=20)
    fig1 = plt.gcf()
    plt.show(block=True)
    plt.draw()
    fig1.savefig(os.path.join(warp_output_folder, str(frame_id) + '_warped.jpg'), dpi=100,
                 bbox_inches='tight', pad_inches=0.3)'''

    return warped_image, inverse_transform


# applies a sliding window approach to the warped road image and finds the left and right lane pixels
def find_lane_pixels_sliding_window(binary_warped, frame_id):
    # compute histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # output image for visualizing the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # get peak points for the left and right halves of the histogram (for windows at the bottom)
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 10               # number of sliding windows for each line
    margin = 100                # width of the sliding windows +/- margin
    minpix = 50                 # minimum number of pixels found to recenter window

    # height of each window
    window_height = np.int(binary_warped.shape[0] // nwindows)

    # get the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # set current positions for the windows (will be updated for each window in nwindows)
    leftx_current = leftx_base
    rightx_current = rightx_base

    # empty lists for holding left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # step through the windows one by one
    for window in range(nwindows):
        # compute window boundaries
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        # get nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # append these indices to the related lane lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # if enough number of pixels are found, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # fit the line pixel positions with polynomials for both image space and world space
    left_fit, right_fit, left_fitx, right_fitx, ploty, out_img, left_fit_cr, right_fit_cr = fit_polynomial(
        binary_warped.shape, leftx, lefty, rightx, righty, out_img, frame_id)

    return left_fit, right_fit, ploty, left_fitx, right_fitx, left_fit_cr, right_fit_cr


# gets left and right lane line pixel positions and fits second order polynomials respectively in image and world spaces
def fit_polynomial(img_shape, leftx, lefty, rightx, righty, out_img, frame_id):
    # fit a second order polynomial to each lane line in image space
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # fit a polynomial to each lane line in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # avoids an error if `left_fit` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    '''# save the result to the output folder
    print('Lane lines are detected for image:', frame_id)
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(os.path.join(sliding_window_output_folder, str(frame_id) + '_sliding_window.jpg'), dpi=100,
                 bbox_inches='tight', pad_inches=0.3)'''

    return left_fit, right_fit, left_fitx, right_fitx, ploty, out_img, left_fit_cr, right_fit_cr


# applies a search approach around a polynomial to the warped road image and finds the left and right lane pixels
def search_around_poly(binary_warped, left_fit, right_fit, frame_id):
    # width of the margin around the previous polynomial to search
    margin = 100

    # get the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # determine the search area using the nonzero pixels within the +/- margin of the polynomial
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                       (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    temp_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # fit the line pixel positions with polynomials for both image space and world space
    left_fit, right_fit, left_fitx, right_fitx, ploty, _, left_fit_cr, right_fit_cr = fit_polynomial(
        binary_warped.shape, leftx, lefty, rightx, righty, temp_img, frame_id)

    '''# save the result to the output folder
    # create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)

    # color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    print('Lane lines are detected for image:', frame_id)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.imshow(result)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig(os.path.join(search_poly_output_folder, str(frame_id) + '_search_poly.jpg'), dpi=100,
                 bbox_inches='tight', pad_inches=0.3)'''

    return left_fit, right_fit, ploty, left_fitx, right_fitx, left_fit_cr, right_fit_cr


# measures the radius of curvature in world space
def measure_curvature_real(ploty, left_fit_cr, right_fit_cr):
    # get the maximum y-value, corresponding to the bottom of the image for the radius of curvature
    y_eval = np.max(ploty)

    # calculate R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad


# computes the vehicle position with respect to the center
def get_vehicle_pos(img_width, left_fitx_base, right_fitx_base):
    img_center = img_width / 2
    lane_center = (left_fitx_base + right_fitx_base) / 2

    return "{:.2f}".format((lane_center - img_center) * xm_per_pix)


# projects bird's eye view of the road back to the original image and shows the related info
def visualize(warped, undist, inv_transform, ploty, left_fitx, right_fitx, pos, curv_left, curv_right, frame_id):
    # create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # warp the blank back to original image space using inverse perspective matrix (inv_transform)
    newwarp = cv2.warpPerspective(color_warp, inv_transform, (image.shape[1], image.shape[0]))

    # combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # hyper-parameters for the information will be printed on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_pos_1 = (5, 30)
    text_pos_2 = (5, 70)
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2

    # vehicle position
    if pos < 0:
        pos_info = 'Vehicle is ' + str(abs(pos)) + 'm right of center'
    elif pos > 0:
        pos_info = 'Vehicle is ' + str(abs(pos)) + 'm left of center'
    else:
        pos_info = 'Vehicle is at center'

    # radius of curvature
    curvature_info = 'Radius of Curvature: Left = ' + str(int(curv_left)) + '(m) Right = ' + str(int(curv_right)) + '(m)'

    # print info messages on the image
    result = cv2.putText(result, curvature_info, text_pos_1, font, font_scale, color, thickness, cv2.LINE_AA)
    result = cv2.putText(result, pos_info, text_pos_2, font, font_scale, color, thickness, cv2.LINE_AA)

    '''# save result image to the output folder
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite(os.path.join(result_output_folder, str(frame_id) + '_result.jpg'), result)'''

    return result


# gets result image list and makes a video
def make_video(img_list, video_path):
    output_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'FMP4'), 25, (img_list[0].shape[1], img_list[0].shape[0]))

    for i in range(len(img_list)):
        output_video.write(img_list[i])

    output_video.release()


# input and output folders
test_input_folder = 'test_images'
undist_output_folder = 'output_images/undistortion'
thresh_output_folder = 'output_images/thresholding'
warp_output_folder = 'output_images/warping'
sliding_window_output_folder = 'output_images/sliding_window'
search_poly_output_folder = 'output_images/search_poly'
result_output_folder = 'output_images/result'


# loads test images in the test_images folder, applies the pipeline and saves resulting images to the related folder
def process_test_images():
    print('Processing test images')

    # (also calibration folder (calib_input_folder) could be used to undistort chessboard images)
    for frame_name in os.listdir(test_input_folder):
        if frame_name.endswith(".jpg"):
            image = cv2.imread(os.path.join(test_input_folder, frame_name))

            # undistort the image
            undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)

            # apply thresholding
            thresholded_image = threshold_image(undistorted_image, frame_name[:-4])

            # apply warping
            warped_img, inv_mat = warp_image(thresholded_image, frame_name[:-4] + '_thresholded')

            # apply sliding window search
            left_fit, right_fit, ploty, left_fitx, right_fitx, left_fit_cr, right_fit_cr = find_lane_pixels_sliding_window(
                warped_img, frame_name[:-4])

            # apply search around polynomial (using the computed polynomials by sliding window approach for illustration)
            left_fit, right_fit, ploty, left_fitx, right_fitx, left_fit_cr, right_fit_cr = search_around_poly(
                warped_img, left_fit, right_fit, frame_name[:-4])

            # compute radius of curvature for left and right lane lines
            left_curverad, right_curverad = measure_curvature_real(ploty, left_fit_cr, right_fit_cr)

            # get the result image
            visualize(warped_img, undistorted_image, inv_mat, ploty, left_fitx, right_fitx,
                      float(get_vehicle_pos(image_size[0], left_fitx[image_size[1] - 1], right_fitx[image_size[1] - 1])),
                      left_curverad, right_curverad, frame_name[:-4])

            '''# save original image and the undistortion result to the output folder
            print('Image is undistorted:', frame_name)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image', fontsize=30)
            ax2.imshow(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
            ax2.set_title('Undistorted Image', fontsize=30)
            fig1 = plt.gcf()
            plt.show(block=True)
            plt.draw()
            fig1.savefig(os.path.join(undist_output_folder, frame_name[:-4] + '_undistorted.jpg'), dpi=100,
                         bbox_inches='tight', pad_inches=0.3)
    
            # save original image and the thresholding result to the output folder
            print('Image is thresholded:', frame_name)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax1.set_title('Original Image', fontsize=20)
            ax2.imshow(thresholded_image, cmap='gray')
            ax2.set_title('Thresholded Image', fontsize=20)
            fig1 = plt.gcf()
            plt.show(block=True)
            plt.draw()
            fig1.savefig(os.path.join(thresh_output_folder, frame_name[:-4] + '_thresholded.jpg'), dpi=100,
                         bbox_inches='tight', pad_inches=0.3)'''


# uncomment this function to apply pipeline on the test images in test_images folder
# also uncomment image saving lines at the end of related function to save results in related folder
#process_test_images()

# variables for lane line pixel positions and polynomials
left_fit = None
right_fit = None
out_img = None
ploty = None
left_fitx = None
right_fitx = None
left_fit_cr = None
right_fit_cr = None
img_list = []

# input video
input_video = cv2.VideoCapture('project_video.mp4')

# process each image of the video
frame_id = 0
while input_video.isOpened():
    ret, frame = input_video.read()

    if not ret:
        break

    print('Processing Frame ', frame_id)

    # undistort the image
    undist_img = cv2.undistort(frame, mtx, dist, None, mtx)

    # apply thresholding
    thres_img = threshold_image(undist_img, frame_id)

    # warp the image
    warped_img, inv_mat = warp_image(thres_img, frame_id)

    # find lane lines by either sliding window approach or polynomial search approach
    if left_fit is None or right_fit is None:
        left_fit, right_fit, ploty, left_fitx, right_fitx, left_fit_cr, right_fit_cr = find_lane_pixels_sliding_window(warped_img, frame_id)
    else:
        # if polynomials are fitted for the previous frame
        left_fit, right_fit, ploty, left_fitx, right_fitx, left_fit_cr, right_fit_cr = search_around_poly(warped_img, left_fit, right_fit, frame_id)

    # compute radius of curvature for both lines
    left_curverad, right_curverad = measure_curvature_real(ploty, left_fit_cr, right_fit_cr)

    # visualize the result and append to the image list tot make a video
    img_list.append(visualize(
        warped_img, undist_img, inv_mat, ploty, left_fitx, right_fitx,
        float(get_vehicle_pos(image_size[0], left_fitx[image_size[1] - 1],
                              right_fitx[image_size[1] - 1])), left_curverad, right_curverad, frame_id))

    frame_id += 1

input_video.release()

# construct the output video
print('Constructing video output')
video_path = 'output_images/project_video_output.mp4'
make_video(img_list, video_path)
print('Video output created:', video_path)
