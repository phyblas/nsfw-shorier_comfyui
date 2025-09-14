import os

import numpy as np
from transformers import pipeline
import torch
import timm
import onnxruntime

from .shori import download_model, torch2pil

class NsfwDetector:
    '''NSFW检测模型基类'''
    def cal_nsfw_score(self, image):
        '''计算NSFW分数'''
        image = image.movedim(-1, 1)
        score = []
        for img_i in image:
            lis_res = self.model(torch2pil(img_i))
            for res in lis_res:
                if res['label'] == 'nsfw':
                    score.append(res['score'])
                    break
        return score

class FalconsaiDetector(NsfwDetector):
    '''Falconsai NSFW检测模型'''
    def __init__(self):
        self.model = pipeline('image-classification', model='Falconsai/nsfw_image_detection')

class AdamcoddDetector(NsfwDetector):
    '''AdamCodd NSFW检测模型'''
    def __init__(self):
        self.model = pipeline('image-classification', model='AdamCodd/vit-base-nsfw-detector')

class UmairrkhnDetector(NsfwDetector):
    '''Umairrkhn NSFW检测模型'''
    def __init__(self):
        self.model = pipeline('image-classification', model='umairrkhn/fine-tuned-nsfw-classification')
    
class MarqoDetector:
    '''Marqo NSFW检测模型'''
    def __init__(self):
        self.model = timm.create_model('hf_hub:Marqo/nsfw-image-detection-384', pretrained=True).eval()
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
    
    def cal_nsfw_score(self, image):
        '''计算NSFW分数'''
        image = image.movedim(-1, 1)
        score = []
        for img_i in image:
            with torch.no_grad():
                img_i = self.transform(torch2pil(img_i)).unsqueeze(0)
                output = self.model(img_i).softmax(dim=-1).cpu()
                score.append(float(output[0,0]))
        return score

model_url = 'https://github.com/iola1999/nsfw-detect-onnx/releases/download/v1.0.0/model.onnx'
model_dir = os.path.join(os.path.dirname(__file__), 'gantman_model')
model_file = os.path.join(model_dir, 'model.onnx')

class GantmanDetector:
    '''GantMan NSFW检测模型'''
    def __init__(self):
        
        if not os.path.exists(model_file):
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            download_model(model_url, model_file, 'GantMan')
        
        self.session = onnxruntime.InferenceSession(model_file)
        self.input_name = self.session.get_inputs()[0].name
    
    def cal_nsfw_score(self, image):
        '''计算NSFW分数'''
        image = image.movedim(-1, 1)
        score = []
        for img_i in image:
            img_i = torch2pil(img_i).resize((299, 299)).convert('RGB')
            img_i = np.array(img_i).astype(np.float32) / 255.0
            img_i = (img_i - 0.5) / 0.5  #  [-1, 1]
            img_i = np.expand_dims(img_i, axis=0)  # NHWC (1, 299, 299, 3)
            output = self.session.run(None, {self.input_name: img_i})[0]
            score.append(float(output[0,[1, 3, 4]].sum()))
        return score