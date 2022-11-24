import numpy as np
import os

# events = np.load('G:\dsesc-20220803T005520Z-001\dsesc\events.npy')
# label_list = np.unique(events[:, 4])
#
# k = 1
# for label in label_list:
#     events_idx = np.argwhere(events[:, 4] == label)
#     if events_idx.shape[0] != 1:
#         np.save('G:/dsesc-20220803T005520Z-001/dsesc/event_slice/' + str(k) + '.npy', events[events_idx].squeeze())
#         k += 1

path = 'G:\dsesc-20220803T005520Z-001\dsesc\event_slice'
files = os.listdir(path)
num = len(files)
print(num)
for file in files:
    print(file)