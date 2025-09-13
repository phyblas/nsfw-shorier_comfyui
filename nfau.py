from transformers import pipeline
import torchvision

torch2pil = torchvision.transforms.ToPILImage()  # Tensor转PIL图像

class NsfwDetector:
    '''NSFW检测模型基类'''
    def cal_nsfw_score(self, image):
        '''计算NSFW分数'''
        image = image.movedim(-1, 1)
        score = []
        for i in range(len(image)):
            lis_res = self.model(torch2pil(image[i]))
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