import os, shutil
import requests
import cv2
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from tqdm import tqdm

# NudeNet检测的对象标签列表
nudenet_label_lis = [
    'female genitalia covered',    # 女性生殖器（覆盖）
    'face female',                 # 女性面部
    'buttocks exposed',            # 暴露的臀部
    'female breast exposed',       # 暴露的女性乳房
    'female genitalia exposed',    # 暴露的女性生殖器
    'male breast exposed',         # 暴露的男性乳房
    'anus exposed',                # 暴露的肛门
    'feet exposed',                # 暴露的脚部
    'belly covered',               # 覆盖的腹部
    'feet covered',                # 覆盖的脚部
    'armpits covered',             # 覆盖的腋下
    'armpits exposed',             # 暴露的腋下
    'face male',                   # 男性面部
    'belly exposed',               # 暴露的腹部
    'male genitalia exposed',      # 暴露的男性生殖器
    'anus covered',                # 覆盖的肛门
    'female breast covered',       # 覆盖的女性乳房
    'buttocks covered'             # 覆盖的臀部
]

# 可能为NSFW的NudeNet检测标签列表
nudenet_nsfw_label_lis = [
    'buttocks exposed',           # 暴露的臀部
    'female breast exposed',      # 暴露的女性乳房
    'female genitalia exposed',   # 暴露的女性生殖器
    'anus exposed',               # 暴露的肛门
    'male genitalia exposed'      # 暴露的男性生殖器
]

# 分辨率与文件映射
reso_file_dic = {320: '320n.onnx', 640: '640m.onnx'}
model_dir = os.path.join(os.path.dirname(__file__), 'notai_model')
nudenet_onnx_file = {res: os.path.join(model_dir, reso_file_dic[res]) for res in reso_file_dic}
repo_url = 'https://huggingface.co/zhangsongbo365/nudenet_onnx/resolve/main/'
nudenet_url = {res: repo_url + reso_file_dic[res] for res in reso_file_dic}

def download_nudenet_model(resolution=320):
    '''下载NudeNet模型文件
    '''
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_url = nudenet_url[resolution]
    model_file = nudenet_onnx_file[resolution]
    try:
        temp_file = model_file + '.download'
        with requests.get(model_url, stream=True) as response:
            text = '正在下载NudeNet模型'
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
        raise RuntimeError(f'NudeNet模型下载失败: {e}') from e

class NudenetDetector:
    '''NudeNet检测模型 - 用于检测图像中的特定内容
    参考:
      - https://github.com/notAI-tech/NudeNet/blob/v3/nudenet/nudenet.py
      - https://github.com/phuvinh010701/ComfyUI-Nudenet/blob/main/Nudenet.py
    '''
    def __init__(self, model_resolution=320):
        self.model_resolution = model_resolution
        model_file = nudenet_onnx_file[model_resolution]
        if not os.path.exists(model_file):
            download_nudenet_model(model_resolution)
        
        self.nudenet_session = onnxruntime.InferenceSession(
            model_file, providers=C.get_available_providers())
    
    def detect_nudenet_part(self, image):
        '''检测图像中的NudeNet部分'''
        detection_lis = []
        for img in image:
            input_name = self.nudenet_session.get_inputs()[0].name
            img, x_pad, y_pad, x_ratio, y_ratio, ori_width, ori_height = self.preprocess(img)
            output = self.nudenet_session.run(None, {input_name: img})
            detection_lis.append(self.postprocess(
                output, x_pad, y_pad, x_ratio, y_ratio, ori_width, ori_height))
        
        return detection_lis
        
    def preprocess(self, img):
        '''预处理图像'''
        img = np.clip(255.0 * img.cpu().numpy(), 0, 255).astype(np.uint8)
        mat_c3 = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    
        max_size = max(mat_c3.shape[:2])  # 获取宽度和高度的最大值
        x_pad = max_size - mat_c3.shape[1]
        x_ratio = max_size / mat_c3.shape[1]
        y_pad = max_size - mat_c3.shape[0]
        y_ratio = max_size / mat_c3.shape[0]
    
        mat_pad = cv2.copyMakeBorder(mat_c3, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT)
    
        input_blob = cv2.dnn.blobFromImage(
            mat_pad,
            1 / 255.0,  # 标准化
            (self.model_resolution, self.model_resolution),  # 调整到模型输入尺寸
            (0, 0, 0),  # 均值减法
            swapRB=True,  # 交换红蓝通道
            crop=False,  # 不裁剪
        )
        
        return (input_blob, x_pad, y_pad, x_ratio, y_ratio, img.shape[1], img.shape[0])
    
    def postprocess(self, output, x_pad, y_pad, x_ratio, y_ratio, ori_width, ori_height):
        '''后处理检测结果'''
        output = np.transpose(np.squeeze(output[0]))
        row_lis = output.shape[0]
        bbox_lis = []
        score_lis = []
        class_id_lis = []
    
        for i in range(row_lis):
            classes_score = output[i][4:]
            max_score = np.amax(classes_score)
    
            if max_score >= 0.2:
                class_id = np.argmax(classes_score)
                x, y, w, h = output[i][0:4]
    
                # 从中心坐标转换为左上角坐标
                x = x - w / 2
                y = y - h / 2
    
                # 缩放到原始图像尺寸
                x = x * (ori_width + x_pad) / self.model_resolution
                y = y * (ori_height + y_pad) / self.model_resolution
                w = w * (ori_width + x_pad) / self.model_resolution
                h = h * (ori_height + y_pad) / self.model_resolution
    
                # 裁剪坐标到图像边界
                x = max(0, min(x, ori_width))
                y = max(0, min(y, ori_height))
                w = min(w, ori_width - x)
                h = min(h, ori_height - y)
    
                class_id_lis.append(class_id)
                score_lis.append(max_score)
                bbox_lis.append([x, y, w, h])
    
        index_lis = cv2.dnn.NMSBoxes(bbox_lis, score_lis, 0.0, 0.45)
    
        # 处理检测到的各个部分
        detection = []
        for i in index_lis:
            bbox = bbox_lis[i]
            score = score_lis[i]
            class_id = class_id_lis[i]
    
            x, y, w, h = bbox
            label = nudenet_label_lis[class_id]
            detection_dic = {
                'label': label,  # 标签
                'score': float(score),  # 分数
                'bbox': [int(x), int(y), int(w), int(h)],  # 边界框
            }
            detection.append(detection_dic)
            
        return detection
    
    def cal_nsfw_score(self, image):
        '''计算NSFW分数'''
        detection = self.detect_nudenet_part(image)
        image = image.movedim(-1, 1)
        score = []
        for detection_i in detection:
            # 找分数最高的元素，只使用这个分数。如果没找到任何元素则返回0
            max_score = 0
            for detection_ii in detection_i:
                if detection_ii['label'] in nudenet_nsfw_label_lis:
                    if detection_ii['score'] > max_score:
                        max_score = detection_ii['score']
            score.append(max_score)
        return score