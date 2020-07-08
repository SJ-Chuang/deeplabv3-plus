import argparse
import numpy as np
import os
import cv2
import json
import time
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import Deeplabv3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=Path, help='Path to original image', required=True)
    parser.add_argument('--mask_path', type=Path, help='Path to masked image', required=True)
    parser.add_argument('--save_dir', type=Path, help='Path to saved results', default='./result')
    parser.add_argument('--input_height', type=int, help='image input height', default=720)
    parser.add_argument('--input_width', type=int, help='image input width', default=1280)
    parser.add_argument('--input_channel', type=int, help='image input channel', default=3)
    parser.add_argument('--classes', type=int, help='number of classes', default=21)
    parser.add_argument('--activation', type=str, help='activation function in output layer', default='softmax')
    parser.add_argument('--val_split', type=float, help='ratio of validation/total dataset', default=0.1)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=8)
    parser.add_argument('--epochs', type=int, help='epoch', default=1000)
    args = parser.parse_args()
    return args

class LabelEncoder:
    def fit(self, images):
        self.classes_ = np.unique(images)
        
    def transform(self, images):
        for i, class_ in enumerate(self.classes_):
            images = np.where(images==class_, i, images)
        return images
    
    def fit_transform(self, images):
        self.fit(images)
        return self.transform(images)

class MeanIoU(tf.keras.metrics.MeanIoU):
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)
        
class DeepLabV3_plus:
    def __init__(self, args):
        self.args = args
        self.image_read()
        self.image_preprocessing()
        input_shape = (args.input_height, args.input_width, args.input_channel)
        self.model = Deeplabv3(
            input_shape=input_shape,
            classes=args.classes,
            backbone='xception',
            activation=args.activation)
        self.model.compile(
            optimizer=optimizers.Adam(lr=args.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=[MeanIoU(num_classes=args.classes)]
        )
        
        X_train = np.array([image for image in self.images.values()])
        y_train = np.array([mask for mask in self.masks.values()])
        
        if not os.path.isdir(args.save_dir): os.mkdir(args.save_dir)
        
        mc = ModelCheckpoint(
            os.path.join(args.save_dir, 'model.tf'),
            verbose=1,
            save_best_only=True,
            save_weight_only=True)
        
        start_time = time.time()
        history = self.model.fit(X_train, y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_split=args.val_split,
            callbacks=[mc]
        )
        end_time = time.time()
        
        print("Total spent time: {}".format(end_time-start_time))
        
        f = open(os.path.join(args.save_dir, 'history.json'), 'w')
        f.write(json.dumps(str(history.history)))
        f.close()
        
    def image_read(self):
        assert os.path.isdir(self.args.image_path), "Not found IMAGES folder from --image_path"
        assert os.path.isdir(self.args.mask_path), "Not found MASKS folder from --mask_path"
        self.images, self.masks = {}, {}
        for image_name in os.listdir(self.args.image_path):
            self.images[image_name] = cv2.imread(os.path.join(self.args.image_path, image_name))
            self.masks[image_name] = cv2.imread(os.path.join(self.args.mask_path, image_name), 0)
    
    def image_preprocessing(self):        
        encoder = LabelEncoder()
        for name, mask in self.masks.items():
            self.masks[name] = encoder.fit_transform(mask)
            
            
            
if __name__ == '__main__':
    DeepLabV3_plus(parse_args())

