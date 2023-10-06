import argparse
from PIL import Image
import numpy as np
import os
 
def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Learn and classify image data.')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    mutex_group = parser.add_mutually_exclusive_group(required=True)
    mutex_group.add_argument('-k', type=int, 
                             help='run k-NN classifier (if k is 0 the code may decide about proper K by itself')
    mutex_group.add_argument("-b", 
                             help="run Naive Bayes classifier", action="store_true")
    parser.add_argument("-o", metavar='filepath', 
                        default='classification.dsv',
                        help="path (including the filename) of the output .dsv file with the results")
    return parser


def diff_of_two_pictures(im1,im2):
    d1 = im1 - im2
    diff = np.linalg.norm(d1)

    return diff

def image_to_array(path_to_pic):
    image = np.array(Image.open(path_to_pic)).flatten()

    return image

def fill_truth(truth_dict, filepath):
    with open(filepath,"r",encoding="utf-8") as infile:
        for line in infile:
            spl=line.split(":")
            truth_dict[spl[0]] = spl[1]

def make_result_average(average_results):
    for letter in average_results:
        shape = average_results[letter][0].shape
        average_result = np.zeros(shape)
        num_of_results =  0
        for result in average_results[letter]:
            average_result = np.add(average_result,result)
            num_of_results +=1
        average_result = np.floor_divide(average_result,num_of_results)
        average_results[letter] = average_result

def classify(test_data, filepath):
    min_diff = 99999999
    img = image_to_array(filepath)

    for category in test_data:
        difference = diff_of_two_pictures(img,test_data[category])
        if difference < min_diff:
            min_diff = difference
            ret = category
    return ret

def write_results(res,test_path):
    with open(test_path,"w",encoding="utf-8") as outfile:
        for answer in res:
            outfile.write(answer)


if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    truth = dict()
    train_images = dict()
    average_result = dict()

    results = list()

    for file in os.listdir(args.train_path):
        if file[-3:] == "dsv":
            fill_truth(truth, os.path.join(args.train_path,file))
        else:
            train_images[file] = image_to_array(os.path.join(args.train_path,file))
    
    for image in train_images:
        if truth[image] in average_result:
            average_result[truth[image]].append(train_images[image])
            continue
        average_result[truth[image]] = []
        average_result[truth[image]].append(train_images[image])
    make_result_average(average_result)

    for file in os.listdir(args.test_path):
        if file[-3:] == "dsv":
            continue
        category = classify(average_result,(os.path.join(args.test_path,file)))

        results.append(file+":"+category)
    write_results(results,args.o)