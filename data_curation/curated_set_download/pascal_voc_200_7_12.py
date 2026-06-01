import os
import shutil

VOC_DIR = "/scratch/akash/VOC/"


def main():
    os.makedirs(VOC_DIR, exist_ok=True)

    years = ['2007', '2012']
    # only tar files that contain the training set (trainval, not test)
    tar_files = {
        '2007': 'VOCtrainval_06-Nov-2007', 
        '2012': 'VOCtrainval_11-May-2012', #contains only 5k images. 
    }
    #NOTE: 1464(Original segmentaion split)+ 7148(SBD/Augmented)= 8648 images
    for year in years:
        tar_file_name = tar_files[year]
        tar_path = os.path.join(VOC_DIR, tar_file_name + '.tar')
        # download the tar file if it is not there
        if not os.path.isfile(tar_path):
            os.system(
                'wget -t0 -c -P ' + VOC_DIR + ' '
                'http://host.robots.ox.ac.uk/pascal/VOC/voc' + year + '/' + tar_file_name + '.tar'
            )
        # extract into VOC_DIR
        os.system('tar -xf ' + tar_path + ' -C ' + VOC_DIR)

    # read only the training set file names for both years
    image_names = []
    no_images = []
    for year in years:
        train_txt = os.path.join(VOC_DIR, 'VOCdevkit', 'VOC' + year, 'ImageSets', 'Main', 'train.txt')
        with open(train_txt) as f:
            names = f.read().splitlines()
            image_names.append(names)
            no_images.append(len(names))

    print(f"VOC 2007 training images: {no_images[0]}")
    print(f"VOC 2012 training images: {no_images[1]}")

    return image_names, no_images


if __name__ == '__main__':
    main()
