import csv
from PIL import Image
import tqdm

position = "hard/position.csv"
prevImage = None
nextImageFlag = True
ff = None

with open(position, "r") as f:
    reader = csv.DictReader(f)
    for line in tqdm.tqdm(reader, total=294334):
        imageName = line['ImageName']
        im: Image.Image = Image.open(f"images/{imageName}", "r")
        w, h = im.size
        if prevImage and prevImage != imageName:
            nextImageFlag = True
            ff.close()
        categoryId = int(line['CategoryId'])
        left = float(line['Left'])
        top = float(line['Top'])
        right = float(line['Right'])
        bottom = float(line['Bottom'])
        if nextImageFlag:
            ff = open(f"labels/{imageName}".replace('.jpg', '.txt'), "w")
            ff.write(f"{categoryId - 1} {left / w} {top / h} {(right-left) / w} {(bottom-top) / h}\n")
            nextImageFlag = False
        else:
            ff.write(f"{categoryId - 1} {left / w} {top / h} {(right-left) / w} {(bottom-top) / h}\n")
        im.close()
        prevImage = imageName
