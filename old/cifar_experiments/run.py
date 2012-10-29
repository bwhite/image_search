import vision_data
import imfeat
import cPickle as pickle

D = vision_data.CIFAR10()
F = imfeat.GIST()


def compute_features(class_name_images):
    out = {}
    for class_name, image in class_name_images:
        out.setdefault(class_name, []).append(F(image))
    return out

pickle.dump(compute_features(D.single_image_class_boxes('train')), open('train.pkl', 'w'), -1)
pickle.dump(compute_features(D.single_image_class_boxes('test')), open('test.pkl', 'w'), -1)
    
