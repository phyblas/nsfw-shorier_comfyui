import os, requests, shutil, re, json
from glob import glob1

from PIL import Image, ImageDraw, ImageFont, ExifTags
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch, torchvision
from torch import nn
from torchvision.transforms import Resize
from transformers import CLIPImageProcessor, CLIPConfig, CLIPVisionModel, PreTrainedModel, pipeline
from tqdm import tqdm

from comfy.cli_args import args
import folder_paths

# 优先使用GPU进行处理
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# 图像转换工具
topil = torchvision.transforms.ToPILImage()  # Tensor转PIL图像
totensor = torchvision.transforms.ToTensor()  # PIL图像转Tensor

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
        with torch.no_grad():
            image_batch = self.vision_model(img.to(device))[1]
            image_embed = self.visual_projection(image_batch)
            sensitivity = 0.14 * 0.5 - 0.1

            special_cos_dist = cal_cos_dist(image_embed, self.special_care_embeds)
            special_score = special_cos_dist - self.special_care_embeds_weights.unsqueeze(0) + sensitivity
            
            cos_dist = cal_cos_dist(image_embed, self.concept_embeds)
            concept_score = cos_dist - self.concept_embeds_weights.unsqueeze(0) + sensitivity
            concept_score += torch.any(special_score > 0, dim=1, keepdim=True) * 0.01
            sigmoid_score = nn.functional.sigmoid(concept_score * 100)
            
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
        '''获取NSFW分数'''
        image = image.movedim(-1, 1)
        clip_input = self.feature_extractor(image, do_rescale=False, return_tensors='pt')
        score = self.safety_checker(clip_input.pixel_values)
        return score

class FalconsaiDetector:
    '''Falconsai NSFW检测模型'''
    def __init__(self):
        self.model = pipeline('image-classification', model='Falconsai/nsfw_image_detection')
    
    def cal_nsfw_score(self, image):
        '''获取NSFW分数'''
        image = image.movedim(-1, 1)
        score = []
        for i in range(len(image)):
            lis_res = self.model(topil(image[i]))
            for res in lis_res:
                if res['label'] == 'nsfw':
                    score.append(res['score'])
                    break
        return score

class AdamcoddDetector(FalconsaiDetector):
    '''AdamCodd NSFW检测模型'''
    def __init__(self):
        self.model = pipeline('image-classification', model='AdamCodd/vit-base-nsfw-detector')

# 模型名称映射字典
model_name_dic = {
    'compvis': CompvisDetector,
    'falconsai': FalconsaiDetector, 
    'adamcodd': AdamcoddDetector,
}
model_name_lis = list(model_name_dic)
model_dic = {}

def hex2rgb(s):
    '''将十六进制颜色代码转换为RGB元组'''
    try:
        s = re.findall('([0-9a-f]{2})'*3, s, flags=re.IGNORECASE)[0]
        return tuple(int(x, 16) for x in s)
    except:
        return (0, 0, 0)  # 默认返回黑色

def get_image_font(font, font_size):
    '''获取字体文件，优先使用TTF或TTC字体'''
    # 在当前包的font文件夹中查找
    dir_font = os.path.join(os.path.dirname(__file__), 'font')
    font_lis = glob1(dir_font, '*')
    if font in font_lis:
        font = os.path.join(dir_font, font)
    else:
        # 在comfyui-textoverlay包的fonts文件夹中查找
        dir_font = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'comfyui-textoverlay', 'fonts')
        font_lis = glob1(dir_font, '*')
        if font in font_lis:
            font = os.path.join(dir_font, font)
    
    return ImageFont.truetype(font, font_size)

def add_text(image, text, font_size, fill_color, stroke_color, stroke_width, font):
    '''
    在图像上添加文字
    参考：https://github.com/munkyfoot/ComfyUI-TextOverlay/blob/main/nodes.py
    '''
    image = topil(image)
    dr = ImageDraw.Draw(image)
    font = get_image_font(font, font_size)
    stroke_width = int(font_size * stroke_width * 0.5)
    
    # 计算文本边界框
    bb = dr.multiline_textbbox(
        xy=(0, 0),
        text=text,
        font=font,
        stroke_width=stroke_width,
        align='center',
        spacing=0,
    )
    text_width = bb[2] - bb[0]
    text_height = bb[3] - bb[1]
    xy = ((image.width - text_width) / 2, (image.height - text_height) / 2)
    
    # 绘制文字
    dr.text(
        xy=xy,
        text=text,
        fill=hex2rgb(fill_color),
        stroke_fill=hex2rgb(stroke_color),
        stroke_width=stroke_width,
        font=font,
        align='center',
        spacing=0,
    )
    return totensor(image)

def resize_and_pad(a1, a2):
    '''调整图像大小并保持宽高比，使用填充'''
    if a1.shape[2] / a2.shape[2] < a1.shape[1] / a2.shape[1]:
        a1 = Resize([a2.shape[1], int(a2.shape[1] * a1.shape[2] / a1.shape[1])])(a1)
    else:
        a1 = Resize([int(a2.shape[2] * a1.shape[1] / a1.shape[2]), a2.shape[2]])(a1)
    
    p1 = a2.shape[1] - a1.shape[1]
    p2 = a2.shape[2] - a1.shape[2]
    return torch.nn.functional.pad(a1, [0, p2, 0, p1], mode='replicate')

### 以下是节点类定义 ###

class GetNsfwScore:
    '''获取NSFW分数 - 输出图像被判定为NSFW的可能性分数'''
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'model_name': (model_name_lis,),
            },
        }
    
    RETURN_TYPES = ('FLOAT',)
    RETURN_NAMES = ('score',)
    FUNCTION = 'get_score'
    CATEGORY = 'NSFW'
    
    def get_score(self, image, model_name):
        if model_name not in model_dic:
            model_dic[model_name] = model_name_dic[model_name]()
        
        model = model_dic[model_name]
        score = model.cal_nsfw_score(image)
        
        return (score,)

class IsNsfw(GetNsfwScore):
    '''NSFW判定 - 根据阈值判断图像是否为NSFW内容'''
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'model_name': (model_name_lis,),
                'threshold': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
            },
        }
    
    RETURN_TYPES = ('BOOLEAN', 'FLOAT')
    RETURN_NAMES = ('NSFW', 'score')
    FUNCTION = 'is_nsfw'
    
    def is_nsfw(self, image, model_name, threshold):
        if threshold == 1:
            return [False] * len(image)
        
        score = self.get_score(image, model_name)[0]
        return (list(np.array(score) > threshold), score)

class ReplaceIfNsfw(IsNsfw):
    '''NSFW内容替换 - 使用替代图像替换被判定为NSFW的图像'''
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'replace_image': ('IMAGE',),
                'model_name': (model_name_lis,),
                'threshold': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'resize': ('BOOLEAN', {'default': False}),
            },
        }
    
    RETURN_TYPES = ('IMAGE', 'BOOLEAN', 'FLOAT')
    RETURN_NAMES = ('image', 'NSFW', 'score')
    FUNCTION = 'replace_nsfw'
    
    def replace_nsfw(self, image, replace_image, model_name, threshold, resize):
        is_nsfw, score = self.is_nsfw(image, model_name, threshold)
        no_nsfw = not any(is_nsfw)
        if no_nsfw:
            return (image, is_nsfw, score)

        all_nsfw = all(is_nsfw)
        image = image.permute(0, 3, 1, 2)
        replace_image = replace_image[0].permute(2, 0, 1)
        
        if resize:
            replace_image = Resize(image.shape[-2:])(replace_image)
        elif not all_nsfw:
            replace_image = resize_and_pad(replace_image, image[0])
        
        image_lis = []
        for i in range(len(image)):
            if is_nsfw[i]:
                image_lis.append(replace_image)
            else:
                image_lis.append(image[i])
        
        out_image = torch.stack(image_lis)
        out_image = out_image.permute(0, 2, 3, 1)
        return (out_image, is_nsfw, score)

# NSFW处理模式列表
mode_lis = ['blur', 'mosaic', 'black', 'white', 'checker', 'mean', 'noise', 'nsfw-chan']

class FilterNsfw(IsNsfw):
    '''NSFW内容过滤 - 对NSFW图像进行各种处理'''
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'model_name': (model_name_lis,),
                'threshold': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'mode': (mode_lis,),
                'resolution': ('INT', {'default': 8, 'min': 2, 'max': 256, 'step': 1}),
            },
        }
    
    RETURN_TYPES = ('IMAGE', 'BOOLEAN', 'FLOAT')
    RETURN_NAMES = ('image', 'NSFW', 'score')
    FUNCTION = 'filter_nsfw'
    
    def filter_nsfw(self, image, model_name, threshold, mode, resolution):
        return self._filter_nsfw(image, model_name, threshold, mode, resolution)
        
    def _filter_nsfw(self, image, model_name, threshold, mode, resolution, text_option=None):
        is_nsfw, score = self.is_nsfw(image, model_name, threshold)
        no_nsfw = not any(is_nsfw)
        if no_nsfw:
            return (image, is_nsfw, score)
        
        image = image.permute(0, 3, 1, 2)
        if mode == 'nsfw-chan':
            nsfw_chan = totensor(Image.open(os.path.join(os.path.dirname(__file__), 'img/nsfw-chan.jpg')))
            nsfw_chan = resize_and_pad(nsfw_chan, image[0])

        image_lis = []
        for i in range(len(image)):
            if is_nsfw[i]:
                if mode == 'blur':  # 模糊处理
                    img_i = Resize([resolution, resolution])(image[i])
                    img_i = Resize(image.shape[-2:])(img_i)
                    
                elif mode == 'mosaic':  # 马赛克处理
                    img_i = Resize([resolution, resolution])(image[i])
                    img_i = Resize(image.shape[-2:], torchvision.transforms.functional.InterpolationMode.NEAREST)(img_i)
                    
                elif mode == 'black':  # 全黑
                    img_i = image[i] * 0
                    
                elif mode == 'white':  # 全白
                    img_i = image[i] * 0 + 1
                    
                elif mode == 'checker':  # 黑白棋盘格
                    img_i = image[i] * 0
                    my, mx = torch.meshgrid(torch.arange(image.shape[-2]), torch.arange(image.shape[-1]))
                    size_x = image.shape[-1] / resolution
                    size_y = image.shape[-2] / resolution
                    mz = ((mx % (size_x * 2) < size_x) != (my % (size_y * 2) < size_y))[None, :, :]
                    img_i += mz
                    
                elif mode == 'mean':  # 使用平均颜色填充
                    img_i = image[i] * 0 + image[i].mean(2).mean(1)[:, None, None]
                    
                elif mode == 'noise':  # 随机噪声
                    img_i = torch.rand([3] + list(image.shape[-2:]))
                    
                elif mode == 'nsfw-chan':  # 使用NSFW酱角色图像
                    img_i = nsfw_chan
                
                # 如果提供了文本选项，添加文字
                if text_option:
                    img_i = add_text(img_i, **text_option)
                
                image_lis.append(img_i)
                
            else:
                image_lis.append(image[i])
        
        out_image = torch.stack(image_lis)
        out_image = out_image.permute(0, 2, 3, 1)
        return (out_image, is_nsfw, score)

class FilterNsfwWithText(FilterNsfw):
    '''NSFW内容过滤并添加文字 - 对NSFW图像进行处理并添加自定义文字'''
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'model_name': (model_name_lis,),
                'threshold': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'mode': (mode_lis,),
                'resolution': ('INT', {'default': 8, 'min': 2, 'max': 256, 'step': 1}),
                'text': ('STRING', {'multiline': True, 'default': 'NSFW'}),
                'font_size': ('INT', {'default': 120, 'min': 1, 'max': 9999, 'step': 1}),
                'fill_color': ('STRING', {'default': '#FFFFFF'}),
                'stroke_color': ('STRING', {'default': '#FF0000'}),
                'stroke_width': ('FLOAT', {'default': 0.2, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
                'font': ('STRING', {'default': 'Arial'}),
            },
        }
    
    def filter_nsfw(self, image, model_name, threshold, mode, resolution, text, font_size, fill_color, stroke_color, stroke_width, font):
        text_option = {
            'text': text,
            'font_size': font_size, 
            'fill_color': fill_color, 
            'stroke_color': stroke_color, 
            'stroke_width': stroke_width,
            'font': font,
        }
        return self._filter_nsfw(image, model_name, threshold, mode, resolution, text_option)

class SaveImageSfw(IsNsfw):
    '''安全保存图像 - 保存图像，NSFW内容将被替换为黑色'''
    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE',),
                'filename_prefix': ('STRING', {'default': 'ComfyUI'}),
                'file_type': (['png', 'jpg', 'webp'], ),
                'no_metadata': ('BOOLEAN', {'default': False}),
                'nsfw_detector_model': (model_name_lis,),
                'threshold': ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
            },
            'hidden': {'prompt': 'PROMPT', 'extra_pnginfo': 'EXTRA_PNGINFO'},
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'save_sfw'
    CATEGORY = 'image'
    
    def save_sfw(self, image, filename_prefix, file_type, no_metadata, nsfw_detector_model, threshold, prompt=None, extra_pnginfo=None):
        # 改编自: https://github.com/Goktug/comfyui-saveimage-plus/blob/main/save_image.py
        is_nsfw, _ = self.is_nsfw(image, nsfw_detector_model, threshold)
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_dir, image[0].shape[1], image[0].shape[0])

        image_out = []
        for i in range(len(image)):
            # NSFW图像转换为黑色
            img_i = topil(image[i].permute(2, 0, 1) * (1 - is_nsfw[i]))
            kwarg_dic = dict()
            
            if file_type == 'png':
                kwarg_dic['compress_level'] = 4
                if not no_metadata and not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text('prompt', json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                    kwarg_dic['pnginfo'] = metadata
            else:
                if file_type == 'webp':
                    kwarg_dic['lossless'] = True
                if not no_metadata and not args.disable_metadata:
                    metadata = {}
                    if prompt is not None:
                        metadata['prompt'] = prompt
                    if extra_pnginfo is not None:
                        metadata.update(extra_pnginfo)
                    exif = img_i.getexif()
                    exif[ExifTags.Base.UserComment] = json.dumps(metadata)
                    kwarg_dic['exif'] = exif.tobytes()

            file = f'{filename}_{counter:05}_.{file_type}'
            img_i.save(os.path.join(full_output_folder, file), **kwarg_dic)
            image_out.append({
                'filename': file,
                'subfolder': subfolder,
                'type': 'output',
            })
            counter += 1
        
        return {'ui': {'images': image_out}}

# 节点类映射
NODE_CLASS_MAPPINGS = {
    'GetNsfwScore': GetNsfwScore,
    'IsNsfw': IsNsfw,
    'ReplaceIfNsfw': ReplaceIfNsfw,
    'FilterNsfw': FilterNsfw,
    'FilterNsfwWithText': FilterNsfwWithText,
    'SaveImageSfw': SaveImageSfw
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    'GetNsfwScore': 'Get NSFW score',
    'IsNsfw': 'Is NSFW',
    'ReplaceIfNsfw': 'Replace if NSFW',
    'FilterNsfw': 'Filter NSFW',
    'FilterNsfwWithText': 'Filter NSFW with text',
    'SaveImageSfw': 'Save Image (SFW)'
}

WEB_DIRECTORY = 'web'