import glob
import os

# Current directory
data_dir = "images/"
# Percentage of images to be used for the test set
percentage_test = 10
# Create and/or truncate train.txt and test.txt
file_train = open('train.txt', 'w')
file_test = open('test.txt', 'w')
# Populate train.txt and test.txt
counter = 1
train_cnt = 0
test_cnt = 0
index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(data_dir + "*.jpg"):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    if counter == index_test:
        counter = 1
        file_test.write("data/" + data_dir + title + '.jpg' + "\n")
        test_cnt += 1
    else:
        file_train.write("data/" + data_dir + title + '.jpg' + "\n")
        counter = counter + 1
        train_cnt += 1
print(f"Train Counter = {train_cnt}")
print(f"Test Counter = {test_cnt}")