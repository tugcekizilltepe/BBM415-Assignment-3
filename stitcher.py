import cv2
import numpy as np
from numpy.core.fromnumeric import shape
import math

def stitcher(images):


    image2 = images[1]
    image3 = images[2]
    # Add padding to allow space around the center image to paste other images.
    image1 = images[0]
    
    # Calculate homography transformation for img2 to img1.
    (M21, pts221, pts121, mask21) = transform(image1,image2)
    
    # Transformed image of img1 and img2 using homography.
    tranformedImage2 = cv2.warpPerspective(image2, M21, (image1.shape[1],image1.shape[0]), dst=image1.copy(),borderMode=cv2.BORDER_CONSTANT)

    # Calculate homography transformation for img3 to img1.
    (M31, pts231, pts131, mask31) = transform(image2,image3)
    tranformedImage3 = cv2.warpPerspective(image3, M31, (image2.shape[1],image2.shape[0]), dst=image2.copy(),borderMode=cv2.BORDER_TRANSPARENT)

    # Combine both transformed images
    # Pass tranformedImage3 as first and tranformedImage2 as second parameter in function because
    # laplacian function blend by taking first half of first image and second hafl of second image.
    output_image = laplacian_blending2(tranformedImage3, tranformedImage2, mask31)
    
    cv2.imwrite("output.jpg", output_image)

    return True
    return True

def laplacian_pyramid_blending(image1, image2, mask, num_levels=10):
    g_image1 = image1.copy()
    g_image2 = image2.copy()
    g_mask = mask.copy()

    gp_image1 = [g_image1]
    gp_image2 = [g_image2]
    gp_mask = [g_mask]


    for i in range(num_levels):
        g_image1 = cv2.pyrDown(g_image1)
        g_image2 = cv2.pyrDown(g_image2)
        g_mask = cv2.pyrDown(g_mask)

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
        l_image1 = np.subtract(gp_image1[i-1], cv2.pyrUp(gp_image1[i] ))
        l_image2 = np.subtract(gp_image2[i-1],  cv2.pyrUp(gp_image2[i]) )
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
        ls_ = cv2.pyrUp(ls_)
        
        print(ls_.shape, LS[i].shape)
        ls_ = cv2.add(ls_, LS[i])

    return ls_


def laplacian_blending(image1, image2, mask, num_levels=10):
    image1_copy = image1.copy()
    image2_copy = image2.copy()
    mask_copy = mask.copy()

    gp_image1 = [image1_copy]
    gp_image2 = [image2_copy]
    gp_mask = [mask_copy]

    for i in range(num_levels):
        image1_copy = cv2.pyrDown(image1_copy)
        gp_image1.append(image1_copy)

        image2_copy = cv2.pyrDown(image2_copy)
        gp_image2.append(image2_copy)

        mask_copy = cv2.pyrDown(mask_copy)
        gp_mask.append(mask_copy)

    image1_copy = gp_image1[5]
    lp_image1 = [image1_copy]
        
    image2_copy = gp_image2[5]
    lp_image2 = [image2_copy]
        
    mask_copy = gp_mask[5]
    lp_mask = [mask_copy] 

    for i in range(num_levels - 1, 0, -1):
        gp_ex = cv2.pyrUp(gp_image1[i])
        lap = cv2.subtract(gp_image1[i-1], gp_ex)
        lp_image1.append(lap)

        gp_ex = cv2.pyrUp(gp_image2[i])
        lap = cv2.subtract(gp_image2[i-1], gp_ex)
        lp_image2.append(lap)
        

        gp_ex = cv2.pyrUp(gp_mask[i])
        lap = cv2.subtract(gp_mask[i-1], gp_ex)
        lp_mask.append(lap)


    n1_n2_pyramid = []
    n = 0
    for image1_lab, image2_lab, mask_lab in zip(lp_image1, lp_image2, lp_mask):
        n += 1
        cols, rows, ch = image1_lab.shape
        laplacian = np.hstack(
            (image1_lab[:, 0:int(cols/2)], image2_lab[:, int(cols/2):]))
        n1_n2_pyramid.append(laplacian)

    reconstruct = n1_n2_pyramid[0]
    for i in range(1, 6):
        reconstruct = cv2.pyrUp(reconstruct)
        reconstruct = cv2.add(n1_n2_pyramid[i], reconstruct)

    return reconstruct

def laplacian_blending2(image1, image2, mask, num_levels=10):
    image1_copy = image1.copy()
    image2_copy = image2.copy()
    mask_copy = mask.copy()

    gp_image1 = [image1_copy]
    gp_image2 = [image2_copy]
    gp_mask = [mask_copy]

    for i in range(num_levels):
        image1_copy = cv2.pyrDown(image1_copy)
        gp_image1.append(image1_copy)

        image2_copy = cv2.pyrDown(image2_copy)
        gp_image2.append(image2_copy)

        mask_copy = cv2.pyrDown(mask_copy)
        gp_mask.append(mask_copy)

    image1_copy = gp_image1[5]
    lp_image1 = [image1_copy]
        
    image2_copy = gp_image2[5]
    lp_image2 = [image2_copy]
        
    mask_copy = gp_mask[5]
    lp_mask = [mask_copy] 

    for i in range(num_levels - 1, 0, -1):
        size = (gp_image1[i - 1].shape[1], gp_image1[i - 1].shape[0])
        gp_ex = cv2.pyrUp(gp_image1[i], dstsize=size)
        lap = cv2.subtract(gp_image1[i-1], gp_ex)
        lp_image1.append(lap)

        size = (gp_image2[i - 1].shape[1], gp_image2[i - 1].shape[0])
        gp_ex = cv2.pyrUp(gp_image2[i], dstsize=size)
        lap = cv2.subtract(gp_image2[i-1], gp_ex)
        lp_image2.append(lap)
        
        size = (gp_mask[i - 1].shape[1], gp_mask[i - 1].shape[0])
        gp_ex = cv2.pyrUp(gp_mask[i], dstsize=size)
        lap = cv2.subtract(gp_mask[i-1], gp_ex)
        lp_mask.append(lap)

    LS = []
    for image1_lab, image2_lab, mask_lab in zip(lp_image1, lp_image2, lp_mask):
        print(image1_lab.shape)
        print(mask_lab.shape)
        print(image2_lab.shape)
        ls = image1_lab * mask_lab + image2_lab * (1.0 - mask_lab)
        LS.append(ls)

    laplacian_top = LS[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(LS) - 1
    for i in range(num_levels):
        size = (LS[i + 1].shape[1], LS[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(LS[i+1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)

    return laplacian_lst[num_levels]

def laplacian_blending3(image1, image2, num_levels = 10):
    image1_copy = image1.copy()
    image2_copy = image2.copy()

    gp_image1 = [image1_copy]
    gp_image2 = [image2_copy]

    for i in range(num_levels):
        image1_copy = cv2.pyrDown(image1_copy)
        gp_image1.append(image1_copy)

        image2_copy = cv2.pyrDown(image2_copy)
        gp_image2.append(image2_copy)


    image1_copy = gp_image1[5]
    lp_image1 = [image1_copy]
        
    image2_copy = gp_image2[5]
    lp_image2 = [image2_copy]


    for i in range(num_levels - 1, 0, -1):
        size = (gp_image1[i - 1].shape[1], gp_image1[i - 1].shape[0])
        gp_ex = cv2.pyrUp(gp_image1[i], dstsize=size)
        lap = cv2.subtract(gp_image1[i-1], gp_ex)
        lp_image1.append(lap)

        size = (gp_image2[i - 1].shape[1], gp_image2[i - 1].shape[0])
        gp_ex = cv2.pyrUp(gp_image2[i], dstsize=size)
        lap = cv2.subtract(gp_image2[i-1], gp_ex)
        lp_image2.append(lap)

    LS = []
    for image1_lab, image2_lab in zip(lp_image1, lp_image2):
        rows,cols, dims = image1_lab.shape
        # Merging first and second half of first and second images respectively at each level in pyramid
        mask1 = np.zeros(image1_lab.shape)
        mask2 = np.zeros(image2_lab.shape)

        mask1[:, 0:math.floor(cols/ 2)] = 1
        mask2[:, math.floor(cols / 2):] = 1

        tmp1 = np.multiply(image1_lab, mask1.astype('float32'))
        tmp2 = np.multiply(image2_lab, mask2.astype('float32'))
        tmp = np.add(tmp1, tmp2)
        
        LS.append(tmp)


    laplacian_top = LS[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(LS) - 1
    for i in range(num_levels):
        size = (LS[i + 1].shape[1], LS[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(LS[i+1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)

    return laplacian_lst[num_levels]

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

    if len(matches) > 4:
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