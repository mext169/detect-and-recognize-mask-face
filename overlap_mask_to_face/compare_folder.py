import os

dir1 = 'D:/Datasets/lfw/lfw_mtcnnpy_160_32'
dir2 = 'D:/Datasets/lfw/lfw_mtcnnpy_160_32_with_mask'

list1 = os.listdir(dir1)
list1.sort()
print(len(list1))
print(list1[1001])
list2 = os.listdir(dir2)
list2.sort()
print(len(list2))
print(list2[1001])


len_list = len(list2)
for i in range(len_list):
    if list1[i] == list2[i]:
        continue
    else:
        print(os.path.join(dir1, list1[i]))
        break

