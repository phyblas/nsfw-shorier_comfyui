import os, shutil
import requests
import torch
from torch import nn
from transformers import CLIPImageProcessor, CLIPConfig, CLIPVisionModel, PreTrainedModel
from tqdm import tqdm

# 优先使用GPU进行处理
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

def cal_cos_dist(a, b):
    '''计算余弦相似度'''
    return nn.functional.cosine_similarity(a.unsqueeze(1), b, dim=-1)

# CompVis模型存储目录
compvis_model_dir = os.path.join(os.path.dirname(__file__), 'compvis_model')

def download_compvis_model():
    '''下载CompVis模型文件'''
    if not os.path.exists(compvis_model_dir):
        os.makedirs(compvis_model_dir)
    
    # 下载模型文件
    root_url = 'https://huggingface.co/CompVis/stable-diffusion-safety-checker/resolve/refs%2Fpr%2F41/'
    model_file = os.path.join(compvis_model_dir, 'model.safetensors')
    model_url = root_url + 'model.safetensors'
    
    try:
        temp_file = model_file + '.download'
        with requests.get(model_url, stream=True) as response:
            text = '正在下载CompVis模型'
            total = int(response.headers.get('content-length', 0))
            pbar = tqdm(None, total=total, unit='b', unit_scale=True, desc=text)
            with open(temp_file, 'wb') as f:
                # 分块下载大文件
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            pbar.close()
        shutil.move(temp_file, model_file)
    except Exception as e:
        raise RuntimeError(f'CompVis模型下载失败: {e}') from e
    
    # 下载配置文件
    for config_json in ['config.json', 'preprocessor_config.json']:
        config_url = root_url + config_json
        config_file = os.path.join(compvis_model_dir, config_json)
        try:
            with requests.get(config_url) as response:
                open(config_file, 'wb').write(response.content)
        except Exception as e:
            raise RuntimeError(f'CompVis配置文件下载失败: {e}') from e

class CLIPSafetyChecker(PreTrainedModel):
    '''
    CompVis使用的NSFW检查器
    改编自:
      - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py
      - https://github.com/Acly/comfyui-tooling-nodes/blob/main/nsfw.py
    '''
    config_class = CLIPConfig
    _no_split_modules = ['CLIPEncoderLayer']

    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        proj_dim = config.projection_dim

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, proj_dim, bias=False)

        self.concept_embeds = nn.Parameter(torch.ones(17, proj_dim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, proj_dim), requires_grad=False)
        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = nn.Parameter(torch.ones(3), requires_grad=False)

    def forward(self, img):
        '''前向传播计算NSFW分数'''
        with torch.no_grad():
            image_batch = self.vision_model(img.to(device))[1]
            image_embed = self.visual_projection(image_batch)
            sensitivity = 0.14 * 0.5 - 0.1

            # 计算特殊关注内容的余弦距离
            special_cos_dist = cal_cos_dist(image_embed, self.special_care_embeds)
            special_score = special_cos_dist - self.special_care_embeds_weights.unsqueeze(0) + sensitivity
            
            # 计算一般概念的余弦距离
            cos_dist = cal_cos_dist(image_embed, self.concept_embeds)
            concept_score = cos_dist - self.concept_embeds_weights.unsqueeze(0) + sensitivity
            concept_score += torch.any(special_score > 0, dim=1, keepdim=True) * 0.01
            
            # 使用sigmoid函数获得0～1的分数值
            sigmoid_score = nn.functional.sigmoid(concept_score * 100)
            # 返回每个图像的最高分数
            return [float(max(cs)) for cs in sigmoid_score]

class CompvisDetector:
    '''CompVis NSFW检测模型'''
    def __init__(self):
        # 检查模型文件是否存在，不存在则下载
        for file_name in ['model.safetensors', 'config.json', 'preprocessor_config.json']:
            model_file = os.path.join(compvis_model_dir, file_name)
            if not os.path.exists(model_file):
                download_compvis_model()
                break
        self.feature_extractor = CLIPImageProcessor.from_pretrained(compvis_model_dir)
        self.safety_checker = CLIPSafetyChecker.from_pretrained(compvis_model_dir).to(device)
    
    def cal_nsfw_score(self, image):
        '''计算图像的NSFW分数'''
        image = image.movedim(-1, 1)  # 调整维度顺序
        clip_input = self.feature_extractor(image, do_rescale=False, return_tensors='pt')
        score = self.safety_checker(clip_input.pixel_values)
        return score