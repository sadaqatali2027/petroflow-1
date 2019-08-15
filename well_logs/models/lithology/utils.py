import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def equalize(image):
    image = image.astype(np.uint8)
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def plot_examples(batch, test=False, mapping=None, reverse_mapping=None, examples=50):
    mapping = dict() if mapping is None else mapping
    reverse_mapping = dict() if reverse_mapping is None else reverse_mapping
    images = np.stack([segment.core_dl / 255 for well in batch.wells for segment in well.iter_level()])
    labels = np.array([mapping.get(
        segment.core_lithology.FORMATION.values[0],
        segment.core_lithology.FORMATION.values[0]
    ) for well in batch.wells for segment in well.iter_level()])
    if test:
        predictions = batch.proba.argmax(axis=1)
    plt.figure(figsize=(15, int(40 * examples / 50)))
    for i, index in enumerate(np.random.choice(len(images), examples, replace=False)):
        plt.subplot(np.ceil(examples / 5).astype(int), 5, i+1)
        plt.imshow(images[index])
        title = 'label: ' + str(labels[index])
        title = title + '\nprediction: ' + str(
            reverse_mapping.get(predictions[index], predictions[index])
        ) if test else title
        color = 'k'
        if test:
            if labels[index] != reverse_mapping.get(predictions[index], predictions[index]):
                color = 'r'
        plt.title(title, color=color)
        plt.xticks([])
        plt.yticks([])
    plt.show()

class LithologyUtils:
    @classmethod
    def get_lithology(self, batch):
        segments = []
        for well in batch.wells:
            segments.extend([segment.core_lithology.FORMATION.values for segment in well.iter_level(level=-1)])
        return segments

    @classmethod
    def get_labels_mapping(self, array, threshold=50):
        array = np.concatenate(array)
        counter = Counter(array)
        name_mapping = {item: item if counter[item] > threshold else 'другое' for item in counter}
        label_mapping = {item: i for i, item in enumerate(np.unique(list(name_mapping.values())))}
        return {item: label_mapping[name_mapping[item]] for item in name_mapping}, name_mapping
    
    @classmethod
    def get_counter(self, array):
        return Counter(np.concatenate(array))
        