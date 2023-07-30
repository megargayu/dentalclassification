from dataloader.dataset import image_datasets

# Due to the way pillow (PIL) loads images, we need to save 
# all of the tensors as a text file for the Android app to read.

total_string = ""
for image, label in image_datasets["test"]:
    total_string += str(label) + ", " + \
                    str(image.flatten().tolist())[1:-1] + "\n"

with open("tensors.txt", "w") as fout:
    fout.write(total_string)
