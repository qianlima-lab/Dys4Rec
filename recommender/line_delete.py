

file1 = open('D:/yanjiusheng/DCSR/dataset/processed_datasets/cd-seq/userTimeRatio/cd_full.txt', 'r')
file2 = open('D:/yanjiusheng/DCSR/dataset/processed_datasets/cd-seq/userTimeRatio/cd_x1.txt', 'w')
for line in file1.readlines():
    data = line.strip('\n')
    file2.writelines(data)
