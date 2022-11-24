import cv2
import numpy as np
from test import gen_discretized_event_volume
from test import gen_event_images
import torch


def draw_matches(img1, kp1, img2, kp2, color=None):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: ndarray [n1, 2]
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: ndarray [n2, 2]
        matches: ndarray [n_match, 2]
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.
    """
    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2

    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 5
    thickness = 1
    if color:
        c = color
    for i in range(len(kp1)):
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color:
            c = np.random.randint(0, 256, 3) if len(img1.shape) == 3 else np.random.randint(0, 256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[i]).astype(int))
        end2 = tuple(np.round(kp2[i]).astype(int) + np.array([img1.shape[1], 0]))
        c = (int(c[0]), int(c[1]), int(c[2]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    return new_img

events = np.load('G:\dsesc-20220803T005520Z-001\dsesc\events.npy')
original_events = np.load('G:\dsesc-20220803T005520Z-001\dsesc\original_events.npy')
frame_min = original_events[:, 4].min()
frame_max = original_events[:, 4].max()

frame_1 = 70
frame_2 = 85
events_slice_frame_1 = events[np.searchsorted(events[:, 5], frame_1):np.searchsorted(events[:, 5], frame_1 + 1), :]
events_slice_frame_2 = events[np.searchsorted(events[:, 5], frame_2):np.searchsorted(events[:, 5], frame_2 + 1), :]
original_slice_frame_1 = original_events[np.searchsorted(original_events[:, 4], frame_1):np.searchsorted(original_events
                                                                                                [:, 4], frame_1 + 1), :]
original_slice_frame_2 = original_events[np.searchsorted(original_events[:, 4], frame_2):np.searchsorted(original_events
                                                                                                [:, 4], frame_2 + 1), :]
events_volume_1 = gen_discretized_event_volume(torch.Tensor(original_slice_frame_1), [20, 480, 640])
events_volume_2 = gen_discretized_event_volume(torch.Tensor(original_slice_frame_2), [20, 480, 640])
event_images = gen_event_images(events_volume_1[None, :, :, :], 'gen')
event_image = event_images['gen_event_time_image'][0].numpy().sum(0)
event_image *= 255. / event_image.max()
event_image = event_image.astype(np.uint8)
event_image_1 = cv2.cvtColor(event_image, cv2.COLOR_GRAY2RGB)
event_images = gen_event_images(events_volume_2[None, :, :, :], 'gen')
event_image = event_images['gen_event_time_image'][0].numpy().sum(0)
event_image *= 255. / event_image.max()
event_image = event_image.astype(np.uint8)
event_image_2 = cv2.cvtColor(event_image, cv2.COLOR_GRAY2RGB)
common, index_1, index_2 = np.intersect1d(events_slice_frame_1[:, 4], events_slice_frame_2[:, 4], assume_unique=False,
                                          return_indices=True)
kp1 = np.zeros((len(common), 2))
kp2 = np.zeros((len(common), 2))
for row in range(len(common)):
    kp1[row, :] = events_slice_frame_1[index_1[row], 0:2]
    kp2[row, :] = events_slice_frame_2[index_2[row], 0:2]
final_image = draw_matches(event_image_1, kp1, event_image_2, kp2)
cv2.imwrite("feature_draw/image.png", final_image)

# upper_bound_original =
# lower_bound_original =