import cv2
import numpy as np

def stitching(images, ratio = 0.75, reproj_treshold=4.0):


    first_image = images[0]
    second_image = images[1]
    
    first_keypoints, first_features = detect_and_describe(first_image)
    second_keypoints, second_features = detect_and_describe(second_image)

    match = match_key_points(first_keypoints, second_keypoints, first_features, second_features, ratio, reproj_treshold)

    if match is None:
        return None
    
    # otherwise, apply a perspective warp to stitch the images
    # together
    (matches, H, status) = match
    result = cv2.warpPerspective(first_image, H,
        (first_image.shape[1] + second_image.shape[1], first_image.shape[0]))
    result[0:second_image.shape[0], 0:second_image.shape[1]] = second_image
    # check to see if the keypoint matches should be visualized

    # return the stitched image
    return result

def detect_and_describe(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect keypoints in the image
    # detect and extract features from the image
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)

    # convert the keypoints from KeyPoint objects to NumPy
    # arrays
    kps = np.float32([kp.pt for kp in kps])
    # return a tuple of keypoints and features
    return (kps, features)

def match_key_points(kpsA, kpsB, featuresA, featuresB,
    ratio, reprojThresh):
    # compute the raw matches and initialize the list of actual
    # matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    # loop over the raw matches
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
		# computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)
        # return the matches along with the homograpy matrix
        # and status of each matched point
        return (matches, H, status)
    # otherwise, no homograpy could be computed
    return None


def stitcher(images):

    # convert images to grayscale

    first_image = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    second_image = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    # get key points and descriptors for first image
    key_points_1, descriptors_1 = sift.detectAndCompute(first_image)
    # get key points and descriptors for second image
    key_points_2, descriptors_2 = sift.detectAndCompute(first_image)

    # convert the KeyPoint objects to arrays
    key_points_1 = np.float32([key_point.pt for key_point in key_points_1])
    key_points_2 = np.float32([key_point.pt for key_point in key_points_2])
    
    matcher = cv2.BFMatcher()
    initial_matches = matcher.knnMatch(descriptors_1,descriptors_2,k=2)

    matches = []
    ratio = 0.75

    for match in initial_matches:
        if len(match) == 2 and match[0].distance < match[1].distance * ratio:
            matches.append((match[0].trainIdx, match[0].queryIdx))



    if len(matches) > 5:
        # construct the two sets of points
        key_points_1 = np.float32([key_points_1[i] for (_, i) in matches])
        key_points_2 = np.float32([key_points_2[i] for (i, _) in matches])

        H, mask = cv2.findHomography(key_points_1, key_points_2, cv2.RANSAC, 5)
    
    else:
        print("Not enough matches are found.")


def Bonus_perspective_warping(img1, img2, img3):

        mask_img1 = np.ones(img1.shape)
        mask_img2 = np.ones(img2.shape)
        mask_img3 = np.ones(img3.shape)

        img1 = cv2.copyMakeBorder(img1,200,200,500,500, cv2.BORDER_CONSTANT)
        mask_img1 = cv2.copyMakeBorder(mask_img1,200,200,500,500, cv2.BORDER_CONSTANT)
        (M, pts1, pts2, mask) = getTransform(img2, img1, 'homography')
        out = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
        out_mask = cv2.warpPerspective(mask_img2, M, (img1.shape[1], img1.shape[0]))
        m = np.ones(out.shape, dtype='float32')
        m[out_mask!=0] = 0
        out = Laplacian_Pyramid_Blending_with_mask(img1, out, m, 4)
        out = np.clip(out, 0, 255)
        out = np.array(out).astype('uint8')
        (M, pts1, pts2, mask) = getTransform(img3, out,'homography')
        out2 = cv2.warpPerspective(img3, M, (out.shape[1], out.shape[0]))
        out2_mask = cv2.warpPerspective(mask_img3, M, (out_mask.shape[1], out_mask.shape[0]))
        m = np.ones(out2_mask.shape, dtype='float32')
        m[out2_mask!=0] = 0
        #m = np.zeros(out2.shape[0], dtype ='float32')
	out2 = Laplacian_Pyramid_Blending_with_mask(out, out2, m, 4)
        out2 = np.clip(out2, 0, 255)
        #final_output = np.zeros((out2.shape[0], out2.shape[1]+8))
        #final_output[:,4:out2.shape[1]+4] = out2

        plt.imshow(out2, cmap='gray')
        plt.show()
        # Write your codes here
        output_image = out2 # This is dummy output, change it to your output

        # Write out the result
        output_name = sys.argv[5] + "output_homography_lpb.png"
        cv2.imwrite(output_name, output_image)

        return True