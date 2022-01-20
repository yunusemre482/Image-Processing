import glob
import cv2
import numpy as np
from tqdm import tqdm


def read_images():
    filenames = glob.glob("images/*jpg")
    images = []

    for filename in sorted(filenames):
        img = cv2.imread(filename)
        img = cv2.resize(img, (640, 480))
        images.append(img)
    H, W, C = np.array(img.shape) * [3, len(filenames), 1]

    return images, len(filenames), H, W, C


def matching_keypoints(src, tgt, nfeatures=1000,method='orb'):
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create(nfeatures=nfeatures, scoreType=cv2.ORB_FAST_SCORE)

    kp1, des1 = descriptor.detectAndCompute(src, None)
    kp2, des2 = descriptor.detectAndCompute(tgt, None)

    # Using BruteForce Matcher to match detected features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Getting best features
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    print(f"Number of matched keypoints  for source and dest => {len(src_pts)}")
    return src_pts, dst_pts


def warpImages(src, homography, imgout, pos_y, pos_x):
    H, W, C = imgout.shape
    src_h, src_w, src_c = src.shape

    # Checking if image needs to be warped or not
    if homography is not None:

        # Calculating net homography
        t = homography
        homography = np.eye(3)
        for i in range(len(t)):
            homography = t[i] @ homography

        # Finding bounding box
        pts = np.array([[0, 0, 1], [src_w, src_h, 1],
                        [src_w, 0, 1], [0, src_h, 1]]).T
        imageBorder = (homography @ pts.reshape(3, -1)).reshape(pts.shape)
        imageBorder /= imageBorder[-1]
        imageBorder = (
                imageBorder + np.array([pos_x, pos_y, 0])[:, np.newaxis]).astype(int)
        h_min, h_max = np.min(imageBorder[1]), np.max(imageBorder[1])
        w_min, w_max = np.min(imageBorder[0]), np.max(imageBorder[0])

        # Filling the bounding box in imgout
        h_inv = np.linalg.inv(homography)
        for i in tqdm(range(h_min, h_max + 1)):
            for j in range(w_min, w_max + 1):

                if (0 <= i < H and 0 <= j < W):
                    # Calculating image cordinates for src
                    u, v = i - pos_y, j - pos_x
                    src_j, src_i, scale = h_inv @ np.array([v, u, 1])
                    src_i, src_j = int(src_i / scale), int(src_j / scale)

                    # Checking if cordinates lie within the image
                    if (0 <= src_i < src_h and 0 <= src_j < src_w):
                        imgout[i, j] = src[src_i, src_j]

    else:
        imgout[pos_y:pos_y + src_h, pos_x:pos_x + src_w] = src

    return imgout, np.sum(imgout, axis=2).astype(bool)


def blendImages(images, masks, n=4):
    assert (images[0].shape[0] % pow(2, n) ==
            0 and images[0].shape[1] % pow(2, n) == 0)

    # Defining dictionaries for various pyramids
    guassianPyramids = {}
    LaplasianPyramids = {}

    H, W, C = images[0].shape

    for i in range(len(images)):

        # Gaussian Pyramids for iamge enhencment
        G = images[i].copy()
        guassianPyramids[i] = [G]
        for _ in range(n):
            G = cv2.pyrDown(G)
            guassianPyramids[i].append(G)

        # Laplacian Pyramids for image enhancment
        LaplasianPyramids[i] = [G]
        for j in range(len(guassianPyramids[i]) - 2, -1, -1):
            G_up = cv2.pyrUp(G) # increase pyRup level for each level recursively
            G = guassianPyramids[i][j]
            L = cv2.subtract(G, G_up)
            LaplasianPyramids[i].append(L)

    # Blending Pyramids
    common_mask = masks[0].copy()
    common_pyramids = [LaplasianPyramids[0][i].copy()
                       for i in range(len(LaplasianPyramids[0]))]

    ls_ = None
    for i in range(1, len(images)):

        # To decide which is left/right
        y1, x1 = np.where(common_mask == 1)
        y2, x2 = np.where(masks[i] == 1)

        if np.max(x1) > np.max(x2):
            left_py = LaplasianPyramids[i]
            right_py = common_pyramids

        else:
            left_py = common_pyramids
            right_py = LaplasianPyramids[i]

        mask_intersection = np.bitwise_and(common_mask, masks[i])

        if True in mask_intersection:
            y, x = np.where(mask_intersection == 1)
            x_min, x_max = np.min(x), np.max(x)
            split = ((x_max - x_min) / 2 + x_min) / W

            LS = []
            for la, lb in zip(left_py, right_py):
                rows, cols, dpt = la.shape
                ls = np.hstack(
                    (la[:, 0:int(split * cols)], lb[:, int(split * cols):]))
                LS.append(ls)

        else:
            LS = []
            for la, lb in zip(left_py, right_py):
                ls = la + lb
                LS.append(ls)

        ls_ = LS[0]
        for j in range(1, n + 1):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, LS[j])

        common_image = ls_
        common_mask = np.sum(common_image.astype(bool), axis=2).astype(bool)
        common_pyramids = LS

    return ls_


if __name__ == '__main__':
    Images, N, H, W, C = read_images() # read images and return all imges  and some fields as array


    # Image Template for final image
    img_f = np.zeros((H, W, C))
    result_image = []
    masks = []

    print(f"\n-- Process of image {N // 2+1} ---")
    img, mask = warpImages(Images[N // 2], None, img_f.copy(), H // 2, W // 2) # wrap first image to two part for stiching

    result_image.append(img)# result image will stored this array
    masks.append(mask)
    leftHomography = []
    rightHomography = []

    for i in range(1, len(Images) // 2 + 1):

        try:
            # right side of panorama image
            print(f"\n-- Process of image {N // 2 + i+1} ---")
            src_points, dst_points = matching_keypoints(Images[N // 2 + i], Images[N // 2 + (i - 1)])

            M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 2, None, 4000)# we can calculate homography matrix with default opencv functions using ransac

            rightHomography.append(M) # add homography to first part
            img, mask = warpImages(Images[N // 2 + i], rightHomography[::-1],
                             img_f.copy(), H // 2, W // 2)

            result_image.append(img)
            masks.append(mask)
        except:
            pass

        try:
            # left
            print(f"\n-- Process of image {N // 2 - i+1} ---")
            src_points, dst_points = matching_keypoints(Images[N // 2 - i], Images[N // 2 - (i - 1)])
            M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 2, None, 4000)# we can calculate homography matrix with default opencv functions using ransac
            leftHomography.append(M)# add homography to second part of image
            img, mask = warpImages(Images[N // 2 - i], leftHomography[::-1],
                             img_f.copy(), H // 2, W // 2)
            result_image.append(img)
            masks.append(mask)
        except:
            pass

    print(f"\n-- Process of Blending  {N} images begin  ---")
    # Blending all the images together
    firstResult = blendImages(result_image, masks)
    mask = np.sum(firstResult, axis=2).astype(bool)

    y_to_yy, x_to_x = np.where(mask == 1)
    x_min, x_max = np.min(x_to_x), np.max(x_to_x)
    y_min, y_max = np.min(y_to_yy), np.max(y_to_yy)

    result = firstResult[y_min:y_max, x_min:x_max]
    cv2.imwrite("panoroma.jpg", result)
    print("Process succesfully completed and output image saved as  panorama.jpg")
