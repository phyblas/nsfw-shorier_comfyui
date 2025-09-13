import os, re, json
from glob import glob1
from PIL import Image, ImageDraw, ImageFont, ExifTags
from PIL.PngImagePlugin import PngInfo
import torch, torchvision
from torchvision.transforms import Resize

from comfy.cli_args import args
import folder_paths

from .nfau import FalconsaiDetector, AdamcoddDetector, UmairrkhnDetector
from .compvis import CompvisDetector
from .notai import NudenetDetector, nudenet_nsfw_label_lis

# 图像转换工具
torch2pil = torchvision.transforms.ToPILImage()  # Tensor转PIL图像
pil2tensor = torchvision.transforms.ToTensor()   # PIL图像转Tensor

# 模型名称映射字典
model_name_dic = {
    'compvis': CompvisDetector,
    'falconsai': FalconsaiDetector,
    'adamcodd': AdamcoddDetector,
    'umairrkhn': UmairrkhnDetector,
    'nudenet (320)': NudenetDetector,
    'nudenet (640)': lambda: NudenetDetector(640),
}

model_name_lis = list(model_name_dic)
model_dic = {}  # 模型实例缓存

def do_filter(img, mode, resolution):
    '''
    对图像进行各种过滤处理
    Args:
        img: 输入图像张量
        mode: 处理模式
        resolution: 处理分辨率
    Returns:
        处理后的图像张量
    '''
    if mode == 'blur':  # 模糊处理
        img = Resize(img.shape[1:])(Resize([resolution, resolution])(img))
        
    elif mode == 'mosaic':  # 马赛克处理
        img = Resize(img.shape[1:], torchvision.transforms.functional.InterpolationMode.NEAREST)(
            Resize([resolution, resolution])(img))
        
    elif mode == 'black':  # 全黑
        img = img * 0
        
    elif mode == 'white':  # 全白
        img = img * 0 + 1
        
    elif mode == 'checker':  # 黑白棋盘格
        img = img * 0
        my, mx = torch.meshgrid(torch.arange(img.shape[-2]), torch.arange(img.shape[-1]))
        size_x = img.shape[-1] / resolution
        size_y = img.shape[-2] / resolution
        mz = ((mx % (size_x * 2) < size_x) != (my % (size_y * 2) < size_y))[None, :, :]
        img += mz
        
    elif mode == 'mean':  # 使用平均颜色填充
        img = img * 0 + img.mean(2).mean(1)[:, None, None]
        
    elif mode == 'noise':  # 随机噪声
        img = torch.rand(img.shape)
        
    elif mode == 'nsfw-chan':  # 使用NSFW酱角色图像
        nsfw_chan = pil2tensor(Image.open(os.path.join(os.path.dirname(__file__), 'img/nsfw-chan.jpg')))
        img = resize_and_pad(nsfw_chan, img)
        
    return img

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
    image = torch2pil(image)
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
    return pil2tensor(image)

def resize_and_pad(a1, a2):
    '''调整图像大小并保持宽高比，使用填充'''
    if a1.shape[2] / a2.shape[2] < a1.shape[1] / a2.shape[1]:
        a1 = Resize([a2.shape[1], int(a2.shape[1] * a1.shape[2] / a1.shape[1])])(a1)
        d = a2.shape[2] - a1.shape[2]
        d_2 = int(d/2)
        pad = [d_2,d-d_2,0,0]
    else:
        a1 = Resize([int(a2.shape[2] * a1.shape[1] / a1.shape[2]), a2.shape[2]])(a1)
        d = a2.shape[1] - a1.shape[1]
        d_2 = int(d/2)
        pad = [0,0,d_2,d-d_2]
    
    return torch.nn.functional.pad(a1, pad, mode='replicate')

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
        '''计算NSFW分数'''
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
        '''判断是否为NSFW内容'''
        score = self.get_score(image, model_name)[0]
        return ([s > threshold for s in score], score)

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
        '''替换NSFW内容'''
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
        '''过滤NSFW内容'''
        return self._filter_nsfw(image, model_name, threshold, mode, resolution)
        
    def _filter_nsfw(self, image, model_name, threshold, mode, resolution, text_option=None):
        '''内部过滤方法'''
        is_nsfw, score = self.is_nsfw(image, model_name, threshold)
        no_nsfw = not any(is_nsfw)
        if no_nsfw:
            return (image, is_nsfw, score)
        
        image = image.permute(0, 3, 1, 2)
        image_lis = []
        for i in range(len(image)):
            if is_nsfw[i]:
                img_i = do_filter(image[i], mode, resolution)
                
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
        '''过滤NSFW内容并添加文字'''
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
        '''安全保存图像'''
        if threshold == 1:
            is_nsfw = [False] * len(image)
        else:
            is_nsfw, _ = self.is_nsfw(image, nsfw_detector_model, threshold)
        
        # 改编自: https://github.com/Goktug/comfyui-saveimage-plus/blob/main/save_image.py
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir, image[0].shape[1], image[0].shape[0])

        image_out = []
        for i in range(len(image)):
            # NSFW图像转换为黑色
            img_i = torch2pil(image[i].permute(2, 0, 1) * (1 - is_nsfw[i]))
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

class FindNudenetPart:
    '''查找NudeNet检测部分 - 使用NudeNet模型检测图像中的特定部分'''
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
                'image': ('IMAGE',),
                'model_resolution': ([320, 640],)
            },
        }
    
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('detection',)
    FUNCTION = 'find_nudenet_part'
    CATEGORY = 'NSFW'
    
    def find_nudenet_part(self, image, model_resolution):
        '''查找NudeNet检测部分'''
        model = f'nudenet ({model_resolution})'
        if model not in model_dic:
            model_dic[model] = NudenetDetector(model_resolution)
        detection = model_dic[model].detect_nudenet_part(image)
        return (detection,)
    
# 定义查找和审查所需的参数
required_find = {
    'image': ('IMAGE',),
    'model_resolution': ([320, 640],)
}
required_censor = {
    **required_find,
    'mode': (mode_lis,),
    'censor_resolution': ('INT', {'default': 8, 'min': 2, 'max': 256, 'step': 1}),
}

# 为每个NSFW标签添加阈值参数
for label in nudenet_nsfw_label_lis:
    required_find[label+' threshold'] = ('FLOAT', {'default': 0.2, 'min': 0.0, 'max': 1.0, 'step': 0.01})
    required_censor[label+' threshold'] = ('FLOAT', {'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.01})

class FindNsfwPart(FindNudenetPart):
    '''查找NSFW部分 - 专门检测NSFW内容'''
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': required_find,}
    
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('detection',)
    FUNCTION = 'find_nsfw_part'
    
    def find_nsfw_part(self, image, model_resolution, **kwarg_dic):
        '''查找NSFW部分'''
        threshold_dic = {label.replace(' threshold', ''): kwarg_dic[label] for label in kwarg_dic}
        detection, = self.find_nudenet_part(image, model_resolution)
        detection_nsfw = []
        for detection_i in detection:
            detection_nsfw_i = []
            for detection_ii in detection_i:
                for label in threshold_dic:
                    if label == detection_ii['label'] and detection_ii['score'] > threshold_dic[label]:
                        detection_nsfw_i.append(detection_ii)
            detection_nsfw.append(detection_nsfw_i)
        return (detection_nsfw,)

class CensorNsfwPart(FindNsfwPart):
    '''审查NSFW部分 - 检测并对NSFW内容进行审查处理'''
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': required_censor,}
    
    RETURN_TYPES = ('IMAGE', 'STRING',)
    RETURN_NAMES = ('image', 'detection',)
    FUNCTION = 'censor_nsfw_part'

    def censor_nsfw_part(self, image, model_resolution, mode, censor_resolution, **kwarg_dic):
        '''审查NSFW部分'''
        detection, = self.find_nsfw_part(image, model_resolution, **kwarg_dic)
        image = image.permute(0, 3, 1, 2)
        img_lis = []
        for i in range(len(image)):
            detection_i = detection[i]
            img_i = image[i] + 0  # 创建副本
            for detection_ii in detection_i:
                x, y, w, h = detection_ii['bbox']
                img_part = img_i[:, y:y+h, x:x+w]
                img_part[:] = do_filter(img_part, mode, censor_resolution)
            
            img_lis.append(img_i)
        out_image = torch.stack(img_lis)
        out_image = out_image.permute(0, 2, 3, 1)
        return (out_image, detection)
        
# 节点类映射
NODE_CLASS_MAPPINGS = {
    'GetNsfwScore': GetNsfwScore,
    'IsNsfw': IsNsfw,
    'ReplaceIfNsfw': ReplaceIfNsfw,
    'FilterNsfw': FilterNsfw,
    'FilterNsfwWithText': FilterNsfwWithText,
    'SaveImageSfw': SaveImageSfw,
    'FindNudenetPart': FindNudenetPart,
    'FindNsfwPart': FindNsfwPart,
    'CensorNsfwPart': CensorNsfwPart,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    'GetNsfwScore': 'Get NSFW score',
    'IsNsfw': 'Is NSFW',
    'ReplaceIfNsfw': 'Replace if NSFW',
    'FilterNsfw': 'Filter NSFW',
    'FilterNsfwWithText': 'Filter NSFW with text',
    'SaveImageSfw': 'Save Image (SFW)',
    'FindNudenetPart': 'Find NudeNet Part',
    'FindNsfwPart': 'Find NSFW Part',
    'CensorNsfwPart': 'Censor NSFW Part',
}

WEB_DIRECTORY = 'web'