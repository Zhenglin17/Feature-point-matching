from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from test import gen_discretized_event_volume
from test import gen_event_images
import cv2
import os
import torch.nn as nn


class MyDataset(Dataset):
    def __init__(self, events_slice, original_events, space_id):
        self.events = events_slice
        self.original_events = original_events
        self.patch_size = (20, 20)
        self.time_trap = 25000
        self.vol_size = [20, 480, 640]
        self.label = space_id
        # self.patches_1, self.patches_2 = self.get_volume_patches_list()
        # self.labels = self.get_labels_list()
        # print(len(self.labels))
        # print(self.labels[2])

    def __getitem__(self, item):
        pairs = self.get_pairs()
        pair = pairs[item]
        feature_center_1, feature_center_2 = self.get_feature_center(pair)
        return self.get_volume_patches(feature_center_1, feature_center_2, item)

    def __len__(self):
        return len(self.get_pairs())

    def get_pairs(self):
        frames = np.unique(self.events[:, 5])
        pairs = frames[np.transpose(np.triu_indices(len(frames), 1))]
        return pairs

    def get_feature_center(self, pair):
        index_1 = np.argwhere(self.events[:, 5] == pair[0])
        index_2 = np.argwhere(self.events[:, 5] == pair[1])
        feature_center_1 = self.events[index_1[0][0], :]
        feature_center_2 = self.events[index_2[0][0], :]
        return feature_center_1, feature_center_2

    def get_volume_patches(self, feature_center_1, feature_center_2, item):
        x_crop = np.ones((20, self.patch_size[0], self.patch_size[1]))
        y_crop = np.ones((20, self.patch_size[0], self.patch_size[1]))
        # if np.logical_and(feature_center_1[0] > 0.5 * self.patch_size[0],
        #                   feature_center_1[0] < 640 - 0.5 * self.patch_size[0]):
        #     if np.logical_and(feature_center_2[0] > 0.5 * self.patch_size[0],
        #                       feature_center_2[0] < 640 - 0.5 * self.patch_size[0]):
        #         if np.logical_and(feature_center_1[1] > 0.5 * self.patch_size[1],
        #                           feature_center_1[1] < 480 - 0.5 * self.patch_size[1]):
        #             if np.logical_and(feature_center_2[1] > 0.5 * self.patch_size[1],
        #                               feature_center_2[1] < 480 - 0.5 * self.patch_size[1]):
        lower_bound_1 = np.searchsorted(self.original_events[:, 2],
                                        feature_center_1[2] - self.time_trap)
        upper_bound_1 = np.searchsorted(self.original_events[:, 2],
                                        feature_center_1[2] + self.time_trap)
        lower_bound_2 = np.searchsorted(self.original_events[:, 2],
                                        feature_center_2[2] - self.time_trap)
        upper_bound_2 = np.searchsorted(self.original_events[:, 2],
                                        feature_center_2[2] + self.time_trap)
        original_events_slice_1 = self.original_events[lower_bound_1:upper_bound_1, :]
        original_events_slice_2 = self.original_events[lower_bound_2:upper_bound_2, :]
        # patches_list_1.append(self.original_events_list[frame_1][idx_1.squeeze()])
        # patches_list_2.append(self.original_events_list[frame_2][idx_2.squeeze()])
        x_ori, x_crop, y_ori, y_crop = self.gen_patches_volume_pairs(original_events_slice_1,
                                                                     original_events_slice_2,
                                                                     feature_center_1,
                                                                     feature_center_2)
        self.draw_event_images(x_ori, y_ori, feature_center_1, feature_center_2, item)
        return torch.Tensor(x_crop), torch.Tensor(y_crop)

    def draw_event_images(self, x_ori, y_ori, event_center_1, event_center_2, index):
        event_images = gen_event_images(x_ori[None, :, :, :], 'gen')
        event_image = event_images['gen_event_time_image'][0].numpy().sum(0)
        event_image *= 255. / event_image.max()
        event_image = event_image.astype(np.uint8)
        event_image = cv2.cvtColor(event_image, cv2.COLOR_GRAY2RGB)
        # print(event_image.shape)
        draw_1 = cv2.rectangle(event_image, (
            int(event_center_1[0] - 0.5 * self.patch_size[0]), int(event_center_1[1] + 0.5 * self.patch_size[1])),
                               (int(event_center_1[0] + 0.5 * self.patch_size[0]),
                                int(event_center_1[1] - 0.5 * self.patch_size[1])), (0, 0, 255), 2)
        event_images = gen_event_images(y_ori[None, :, :, :], 'gen')
        event_image = event_images['gen_event_time_image'][0].numpy().sum(0)
        event_image *= 255. / event_image.max()
        event_image = event_image.astype(np.uint8)
        event_image = cv2.cvtColor(event_image, cv2.COLOR_GRAY2RGB)
        draw_2 = cv2.rectangle(event_image, (
            int(event_center_2[0] - 0.5 * self.patch_size[0]), int(event_center_2[1] + 0.5 * self.patch_size[1])),
                               (int(event_center_2[0] + 0.5 * self.patch_size[0]),
                                int(event_center_2[1] - 0.5 * self.patch_size[1])), (0, 0, 255), 2)
        draw = np.concatenate((draw_1, draw_2), axis=1)
        # print(draw.shape)
        cv2.imwrite('images_compare/' + 'label_' + str(self.label) + '_' + str(index) + '_simulated_event_compare.png',
                    draw)

    def gen_patches_volume_pairs(self, events_1, events_2, event_center_1, event_center_2):
        events_volume_1 = gen_discretized_event_volume(torch.Tensor(events_1), self.vol_size)
        events_volume_2 = gen_discretized_event_volume(torch.Tensor(events_2), self.vol_size)
        volume_1 = torch.zeros((events_volume_1.shape[0], events_volume_1.shape[1] + self.patch_size[0],
                                events_volume_1.shape[2] + self.patch_size[1]))
        volume_2 = torch.zeros((events_volume_1.shape[0], events_volume_1.shape[1] + self.patch_size[0],
                                events_volume_1.shape[2] + self.patch_size[1]))
        # print(events_volume_1.shape, volume_1.shape)
        volume_1[:, int(0.5 * self.patch_size[0]):int(0.5 * self.patch_size[0] + 480), int(0.5 * self.patch_size[1]):int(0.5 * self.patch_size[1] + 640)] = events_volume_1
        volume_2[:, int(0.5 * self.patch_size[0]):int(0.5 * self.patch_size[0] + 480), int(0.5 * self.patch_size[1]):int(0.5 * self.patch_size[1] + 640)] = events_volume_2
        event_volume_croped_1 = volume_1[:, int(event_center_1[1]):int(event_center_1[1] + self.patch_size[1]),
                                         int(event_center_1[0]):int(event_center_1[0] + self.patch_size[0])]
        event_volume_croped_2 = volume_2[:, int(event_center_2[1]):int(event_center_2[1] + self.patch_size[1]),
                                         int(event_center_2[0]):int(event_center_2[0] + self.patch_size[0])]
        # print(event_volume_croped_1.shape, event_volume_croped_2.shape)
        # if event_volume_croped_1.shape[1] == 0:
        #     print(event_center_2[0] - 0.5 * self.patch_size[0], event_center_2[0] + 0.5 * self.patch_size[0])
        return events_volume_1, event_volume_croped_1, events_volume_2, event_volume_croped_2


# label_list = np.unique(events[:, 4])
# print(label_list[0:10])
# # print(label_list[-1])
# for label in label_list:
#     events_idx = np.argwhere(events[:, 4] == label)
#     events_slice = events[events_idx].squeeze()
#     # print(events_slice)
#     if events_idx.shape[0] >= 2:
#         if len(np.unique(events_slice[:, 5])) >= 2:
#             myDataset = MyDataset(events_slice, original_events, label)
#             myDataloader = DataLoader(myDataset, batch_size=1)
#             for epoch in range(2):
#                 for data_1, data_2 in myDataloader:
#                     print(data_1.shape)

# events_idx = np.argwhere(events[:, 4] == 93237)
# events_slice = events[events_idx].squeeze()
# myDataset = MyDataset(events_slice, original_events, 93237)
# myDataloader = DataLoader(myDataset, batch_size=1)
# for epoch in range(2):
#     for data_1, data_2 in myDataloader:
#         print(next)


def get_and_concat_datasets(original_events):
    ds_list = []
    path = 'G:/dsesc-20220803T005520Z-001/dsesc/event_slice'
    files = os.listdir(path)
    k = 0
    for file in files:
        event_slice = np.load('G:/dsesc-20220803T005520Z-001/dsesc/event_slice/' + str(file))
        label = event_slice[0, -1]
        ds_list.append(MyDataset(events_slice=event_slice, original_events=original_events,
                                 space_id=label))
        print(k)
        k += 1
        if k > 20:
            break
    ds = torch.utils.data.ConcatDataset(ds_list)
    return ds


def get_dataloader():
    events = np.load('G:\dsesc-20220803T005520Z-001\dsesc\events.npy')
    original_events = np.load('G:\dsesc-20220803T005520Z-001\dsesc\original_events.npy')
    ds = get_and_concat_datasets(original_events)
    myDataloader = DataLoader(ds, batch_size=64, shuffle=False, drop_last=True)
    return myDataloader
