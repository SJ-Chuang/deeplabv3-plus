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
    parser.add_argument('--image_path', type=Path, help='Path to original image', default=None)
    parser.add_argument('--mask_path', type=Path, help='Path to masked image', default=None)
    parser.add_argument('--save_dir', type=Path, help='Path to saved results', default='./result')
    parser.add_argument('--input_height', type=int, help='image input height', default=720)
    parser.add_argument('--input_width', type=int, help='image input width', default=1280)
    parser.add_argument('--input_channel', type=int, help='image input channel', default=3)
    parser.add_argument('--classes', type=int, help='number of classes', default=3)
    parser.add_argument('--activation', type=str, help='activation function in output layer', default='softmax')
    parser.add_argument('--val_split', type=float, help='ratio of validation/total dataset', default=0.1)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=4)
    parser.add_argument('--epochs', type=int, help='epoch', default=1000)
    parser.add_argument('--test_path', type=Path, help='Path to test image', default=None)
    parser.add_argument('--model_path', type=str, help='Path to trained model', default=None)
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
        self.image_read()
        if self.args.image_path and self.args.mask_path:
            self.train()
            
        if self.args.test_path and self.args.model_path:
            self.test()
        
    def train(self):    
        self.image_preprocessing()
        X_train = np.array([image for image in self.images.values()])
        y_train = np.array([mask for mask in self.masks.values()])
        
        if not os.path.isdir(self.args.save_dir): os.mkdir(self.args.save_dir)
        
        mc = ModelCheckpoint(
            os.path.join(self.args.save_dir, 'model.h5'),
            verbose=1,
            save_best_only=True,
            save_weight_only=True)
        
        start_time = time.time()
        history = self.model.fit(X_train, y_train,
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            validation_split=self.args.val_split,
            callbacks=[mc]
        )
        end_time = time.time()
        
        print("Total spent time: {}".format(end_time-start_time))
        
        f = open(os.path.join(self.args.save_dir, 'history.json'), 'w')
        f.write(json.dumps(str(history.history)))
        f.close()
    
    def test(self):       
        self.model.load_weights(self.args.model_path)
        classes = self.model.output_shape[-1]
        colors = []
        for c in range(classes):
            color = list(np.random.choice(range(256), size=3))
            colors.append([int(color[0]), int(color[1]), int(color[2])])
            
        for name, image in zip(self.tests.keys(), self.tests.values()):
            seg = self.model.predict(np.array([image]))
            classes = seg.shape[-1]
            seg = np.argmax(seg, axis=-1)[0]            
            img = np.stack((seg,)*3, axis=-1)
            for c in range(1, classes):
                img = np.where(img==[c, c, c], colors[c], img)
                cv2.imwrite(os.path.join(self.args.save_dir, name), img)
    
    def image_read(self):
        if self.args.image_path and self.args.mask_path:
            assert os.path.isdir(self.args.image_path), "Not found IMAGES folder from --image_path"
            assert os.path.isdir(self.args.mask_path), "Not found MASKS folder from --mask_path"        
            self.images, self.masks = {}, {}
            for image_name in os.listdir(self.args.image_path):
                self.images[image_name] = cv2.imread(os.path.join(self.args.image_path, image_name))
                self.masks[image_name] = cv2.imread(os.path.join(self.args.mask_path, image_name), 0)
        if self.args.test_path and self.args.model_path:
            assert os.path.isdir(self.args.test_path), "Not found TESTS folder from --test_path"
            self.tests = {}
            for image_name in os.listdir(self.args.test_path):
                self.tests[image_name] = cv2.imread(os.path.join(self.args.test_path, image_name))
    
    def image_preprocessing(self):        
        encoder = LabelEncoder()
        for name, mask in self.masks.items():
            self.masks[name] = encoder.fit_transform(mask)
            
            
            
if __name__ == '__main__':
    DeepLabV3_plus(parse_args())
    
    
        

