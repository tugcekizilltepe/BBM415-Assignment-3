import cv2
import numpy as np


def stitcher(images):

    image1 = cv2.copyMakeBorder(images[0],300,300,600,600, cv2.BORDER_CONSTANT)
    image2 = images[1]
    image3 = images[2]
    (homography_1, key_points_1, key_points_2, mask_1) = transform(images[2], image1)
    (homography_2, key_points_2, key_points_3, mask_2) = transform(images[1], image1)
    
    m = np.ones_like(image3, dtype='float32')
    m1 = np.ones_like(image2, dtype='float32')

    out1 = cv2.warpPerspective(image3, homography_1, (image1.shape[1],image1.shape[0]))
    out2 = cv2.warpPerspective(image2, homography_2, (image1.shape[1],image1.shape[0]))
    out3 = cv2.warpPerspective(m, homography_1, (image1.shape[1],image1.shape[0]))
    out4 = cv2.warpPerspective(m1, homography_2, (image1.shape[1],image1.shape[0]))

    lpb = laplacian_pyramid_blending(out1,image1,out3)

    lpb1 = laplacian_pyramid_blending(out2,lpb,out4)
    cv2.imwrite('output_homography3_lpb.png',lpb1)


    return True

def laplacian_pyramid_blending(image1, image2, mask, num_levels=10):
    g_image1 = image1.copy()
    g_image2 = image2.copy()
    g_mask = mask.copy()

    gp_image1 = [g_image1]
    gp_image2 = [g_image2]
    gp_mask = [g_mask]


    for i in range(num_levels):
        g_image1 = blur_downsample(g_image1)
        g_image2 = blur_downsample(g_image2)
        g_mask = blur_downsample(g_mask)

        gp_image1.append(np.float32(g_image1))
        gp_image2.append(np.float32(g_image2))
        gp_mask.append(np.float32(g_mask))

# generate Laplacian Pyramids for A,B and masks
    lp_image1  = [gp_image1[num_levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lp_image2  = [gp_image2[num_levels-1]]
    g_mask_lp = [gp_mask[num_levels-1]]

    for i in range(num_levels-1,0,-1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        l_image1 = np.subtract(gp_image1[i-1], upsample(gp_image1[i], (gp_image1[i-1].shape[1],gp_image1[i-1].shape[0]))  )
        l_image2 = np.subtract(gp_image2[i-1], upsample(gp_image2[i] ,(gp_image1[i-1].shape[1],gp_image1[i-1].shape[0])) )
        lp_image1.append(l_image1)
        lp_image2.append(l_image2)
        g_mask_lp.append(gp_mask[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la,lb,gm in zip(lp_image1,lp_image2,g_mask_lp):
        ls = la * gm + lb * (1 - gm)
        LS.append(ls)

    
    # now reconstruct
    ls_ = LS[0]
    for i in range(1,num_levels):
        print(i)
        shape = (LS[i].shape[1], LS[i].shape[0])
        ls_ = upsample(ls_, shape)
        
        print(ls_.shape, LS[i].shape)
        ls_ = cv2.add(ls_, LS[i])

    return ls_

def transform(image1, image2):
        # convert images to grayscale

    sift = cv2.xfeatures2d.SIFT_create()

    # get key points and descriptors for first image
    key_points_1, descriptors_1 = sift.detectAndCompute(image1, None)
    # get key points and descriptors for second image
    key_points_2, descriptors_2 = sift.detectAndCompute(image2, None)

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

    if len(matches) > 10:
        # construct the two sets of points
        key_points_1 = np.float32([key_points_1[i] for (_, i) in matches])
        key_points_2 = np.float32([key_points_2[i] for (i, _) in matches])

        H, mask = cv2.findHomography(key_points_1, key_points_2, cv2.RANSAC, 5)

        return H, key_points_1, key_points_2, mask

    else:
        print("Not enough matches are found.")

def blur_downsample(image):
    image = cv2.GaussianBlur(image, (5,5), 2)
    ratio = 0.5
    downsampled_image = cv2.resize(image, # original image
                        (0,0), # set fx and fy, not the final size
                        fx=ratio, 
                        fy=ratio, 
                        interpolation=cv2.INTER_LINEAR)

    return downsampled_image

def upsample(image, size):

    ratio = 2
    upsampled_image = cv2.resize(image, # original image
                        size, # set fx and fy, not the final size
                        fx=ratio, 
                        fy=ratio, 
                        interpolation=cv2.INTER_LINEAR)

    return upsampled_image

def upsample2(image):
    ratio = 2
    upsampled_image = cv2.resize(image, # original image
                        (0,0), # set fx and fy, not the final size
                        fx=ratio, 
                        fy=ratio, 
                        interpolation=cv2.INTER_LINEAR)

    return upsampled_image