import os, re, shutil
from glob import glob1

import requests
from PIL import Image, ImageDraw, ImageFont
import torch, torchvision
from torchvision.transforms import Resize
from tqdm import tqdm

# 图像转换工具
torch2pil = torchvision.transforms.ToPILImage()  # Tensor转PIL图像
pil2tensor = torchvision.transforms.ToTensor()   # PIL图像转Tensor

def download_model(model_url, model_file, model_name=''):
    try:
        temp_file = model_file + '.download'
        with requests.get(model_url, stream=True) as response:
            text = f'正在下载{model_name}模型'
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
        raise RuntimeError(f'{model_name}模型下载失败: {e}') from e

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