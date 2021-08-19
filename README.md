
#### **Udacity Self-Driving Car Engineer Nanodegree Program**
### **Advanced Lane Finding Project**

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image_corner_1]: ./output_images/calibration/calibration8_corners.jpg "Corner example 1"
[image_corner_3]: ./output_images/calibration/calibration3_corners.jpg "Corner example 2"

[image_undistortion_1]: ./output_images/undistortion/calibration1_undistorted.jpg "Undistortion example 1"
[image_undistortion_2]: ./output_images/undistortion/test6_undistorted.jpg "Undistortion example 2"

[image_thresholding_1]: ./output_images/thresholding/test6_thresholded.jpg "Thresholding example 1"
[image_thresholding_2]: ./output_images/thresholding/straight_lines2_thresholded.jpg "Thresholding example 2"

[image_warping_1]: ./output_images/warping/test6_warped.jpg "Warping example 1"
[image_warping_2]: ./output_images/warping/straight_lines2_warped.jpg "Warping example 2"

[image_warping_thres_1]: ./output_images/warping/test6_thresholded_warped.jpg "Warping threholded example 1"
[image_warping_thres_2]: ./output_images/warping/straight_lines2_thresholded_warped.jpg "Warping threholded example 2"

[image_histogram]: ./output_images/sliding_window/histogram.png "Histogram"

[image_sliding_window_1]: ./output_images/sliding_window/test6_sliding_window.jpg "Sliding window example 1"
[image_sliding_window_2]: ./output_images/sliding_window/straight_lines2_sliding_window.jpg "Sliding window example 2"

[image_search_poly_1]: ./output_images/search_poly/test6_search_poly.jpg "Search polynomial example 1"
[image_search_poly_2]: ./output_images/search_poly/straight_lines2_search_poly.jpg "Search polynomial example 2"

[image_result_1]: ./output_images/result/test6_result.jpg "Result example 1"
[image_result_2]: ./output_images/result/straight_lines2_result.jpg "Result example 2"

### The project consists of the following files:

- **Project.py**

- **project_writeup.md**

- **project_video_output.mp4** and the other images in the **output_images** folder

  

### Camera Calibration

Firstly, the chessboard corners are needed to be located on the calibration images in the **camera_cal** folder. The lists for holding object points and the image points are defined for the 3D world coordinates and the 2D image coordinates of the corners respectively. Each calibration image is converted to grayscale and `cv2.findChessboardCorners()` function is used to find the corners. The images showing the chessboard corners are saved in **output_images/calibration** folder. Below are some examples:

![alt text][image_corner_1]  ![alt text][image_corner_3]



Utilizing the found chessboard corners, `cv2.calibrateCamera()` function is used to compute the camera calibration and distortion coefficients. After that, image undistortion is performed through using the `cv2.undistort()` function by applying the coefficients found. 
![alt text][image_undistortion_1]  

### Pipeline (single images)

#### 1. Distortion correction

Each incoming frame from the camera is undistorted by applying the procedure explained above. An example is shown below:
![alt text][image_undistortion_2]  

#### 2. Image thresholding

In order to get a thresholded binary image to detect the lanes on the road, 4 different thresholding methods are applied. As it can be seen in the `threshold_image()` function, two color thresholds are applied to **R** (in RGB color space as default) and **S** (after converting the image to HLS color space) channels. 

```python
# get R channel in RGB color space
r_channel = image[:, :, 2]

# convert image to HLS color space and get the S and L channels
hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
s_channel = hls_image[:, :, 2]
l_channel = hls_image[:, :, 1]
```

Moreover, a 3x3 **Sobel** filter is applied on the **L** channel in x and y directions, and the results are thresholded.

```python
# apply Sobel to L channel in x direction
sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
abs_sobelx = np.absolute(sobelx)
scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

# apply Sobel to L channel in y direction
sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)
abs_sobely = np.absolute(sobely)
scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))
```

Finally, the thresholded 4 images are combined in a binary image. Sobel x-y thresholds and color (S-R) thresholds are combined separately. 

```python
combined_binary_image[((sybinary == 1) & (sxbinary == 1)) | ((r_binary == 1) & (s_binary == 1))] = 1
```

![alt text][image_thresholding_1]  ![alt text][image_thresholding_2] 


#### 3. Perspective transform

In order to get a bird's eye view of the road, a function named `warp_image()` is developed, which applies perspective transform to the image. The function firstly determines the coordinates of the source points in the original image and the corresponding target points in the bird's eye image. These points are chosen manually by analyzing different test images. 

```python
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
```

This resulted in the following source and destination points:

|  Source   | Destination |
| :-------: | :---------: |
| 585, 455  |   330, 0    |
| 190, 720  |  330, 720   |
| 1115, 720 |  970, 720   |
| 696, 455  |   970, 0    |

After applying `cv2.getPerspectiveTransform()` function to get the transformation, and then using `cv2.warpPerspective()` to warp the image, the resulting bird's eye view of the road images are obtained. Below are some examples of this procedure for both undistorted images and the thresholded versions:
![alt text][image_warping_1] ![alt text][image_warping_thres_1] ![alt text][image_warping_2]  ![alt text][image_warping_thres_2] 

#### 4. Finding the lane lines using histogram peaks with a sliding window approach and performing more efficient line search around previously computed polynomials
In this step, histogram of the bottom half of the warped binary image is computed firstly. Then the two peak points in the histogram are used as the starting search locations of left and right lane line pixels. 

```python
histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
```

An example histogram is shown below, which demonstrates the corresponding peak points of the left and right lane lines.

![alt text][image_histogram]

Two windows are located on the starting search locations (peak points) for the left and right lane lines respectively. 

```python
# get peak points for the left and right halves of the histogram (for windows at the bottom)
midpoint = np.int(histogram.shape[0] // 2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```

The hyper parameters for the windows are:

```python
nwindows = 10               # number of sliding windows for each line
margin = 100                # width of the sliding windows +/- margin
minpix = 50                 # minimum number of pixels found to recenter window
```

The lane line pixels are detected using a sliding window approach by starting from the bottom of the image (peak points) to the top through re-centering windows in each iteration according to the mean points of nonzero pixels. After finding all line pixels, two second order polynomials are fitted for the left and right lane line pixel positions respectively. This approach is implemented in `find_lane_pixels_sliding_window()` and `fit_polynomial()` functions. Below are the results of the sliding window approach for the sample images demonstrated above (the yellow lines show the fitted polynomials):
![alt text][image_sliding_window_1] ![alt text][image_sliding_window_2]

For each incoming frame, computing histogram and finding peak points for the starting search position and applying the same sliding window approach might be time consuming. Therefore, a more efficient approach that uses the lane line positions of the previous frame is applied. For each new frame, the fitted left and right lane polynomials of the previous frame is used to search the lane line pixels nearby. As it can be seen in the `search_around_poly()` function, the line search area is determined according to a margin near the fitted polynomial.

```python
# determine the search area using the nonzero pixels within the +/- margin of the polynomial
left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
```
The resulting images for the same samples are below:
![alt text][image_search_poly_1] ![alt text][image_search_poly_2]

#### 5. Computing the radius of curvature of the lane and the position of the vehicle

The radius of curvature of a lane line is computed by utilizing the fitted polynomial. A detailed mathematical background of the computation could be found [here](https://www.intmath.com/applications-differentiation/8-radius-curvature.php). In this manner, `ym_per_pix` and `xm_per_pix` variables are used as scaling factors for transformation from image space to the real world space:

```python
# scaling factors for pixel to real world distance transformation
ym_per_pix = 30/720     # meters per pixel in y dimension
xm_per_pix = 3.7/700    # meters per pixel in x dimension
```

In the project, `measure_curvature_real()` function does the necessary computations through using the fitted polynomials in world space (`left_fit_cr` and `right_fit_cr`).

```python
# get the maximum y-value, corresponding to the bottom of the image for the radius of curvature
y_eval = np.max(ploty)

# calculate R_curve (radius of curvature)
left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
```

The position of the vehicle is computed by `get_vehicle_pos()` function with respect to the center of the image. The lane center is computed by calculating the distance between left and right lane line pixels at the bottom of the image. Considering the camera is mounted at the center of the car, the vehicle position is the distance between the image center and the lane center.

```python
# computes the vehicle position with respect to the center
def get_vehicle_pos(img_width, left_fitx_base, right_fitx_base):
    img_center = img_width / 2
    lane_center = (left_fitx_base + right_fitx_base) / 2

    return "{:.2f}".format((lane_center - img_center) * xm_per_pix)
```
#### 6. The result showing the lane area and related information

The resulting images demonstrating the lane, radius of curvature, and vehicle position for the same samples are below:

![alt text][image_result_1] 

![alt text][image_result_2]



### Pipeline (video)

A video result of the pipeline can be seen [here](./output_images/project_video_output.mp4) 

---

### Discussion
1. In more challenging environments not containing clearly visible lane lines, the approach might produce less robust results. A more effective thresholding approach could be developed for handling such situations. In this project, magnitude of the gradient in the x and y directions are utilized in addition to color thresholds. In order to make the thresholding process more robust, gradient direction might be utilized. 
2. In this implementation, the road has been considered having a flat shape and all the transformation calculations have been carried out according to this consideration. However, that is not the case in real world and thus the computed lane line curvatures and the vehicle position might have bigger error rates than expected. This could be figured out by analyzing the geometrical shape of the road in the pipeline and making revisions in the implementation.
