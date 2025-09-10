# nsfw-shorier_comfyui

（→ [english description](https://github-com.translate.goog/phyblas/nsfw-shorier_comfyui/blob/master/README.md?_x_tr_sl=zh-CN&_x_tr_tl=en&_x_tr_hl=zh-CN&_x_tr_pto=wapp)）
（→ [日本語の説明](README_日本語.md)）

这个包用于在 ComfyUI 中处理包含 NSFW（Not Safe For Work，不适宜在工作场所观看）内容的图片。其名称来源于日语中的“処理”（shori，意为“处理”）。
## 安装

将此仓库放置在 ComfyUI 的 `ComfyUI/custom_nodes/` 文件夹中。也可以通过 [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) 进行安装。

## 使用的模型

可以选择用于判断 NSFW 内容的模型：
- [compvis](https://huggingface.co/CompVis/stable-diffusion-safety-checker)
- [falconsai](https://huggingface.co/Falconsai/nsfw_image_detection)
- [adamcodd](https://huggingface.co/AdamCodd/vit-base-nsfw-detector)

compvis 是默认模型，根据我的经验，它能提供最准确的结果。但也可以尝试其他模型，请选择自己认为效果最好的一个。

在首次使用时，系统会自动下载模型文件，请耐心等待片刻。之后再次使用即可快速加载，无需重复下载。

## 节点

这个包提供了 6 个节点。

### - GetNsfwScore

这是最基本的节点，仅用于输出表示 NSFW 可能性的分数。分数范围在 0 到 1 之间，分数越高，越有可能是 NSFW 内容。

不同模型的判断标准可能不同，有时会给出差异较大的分数。

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_GetNsfwScore.jpg)

### - IsNsfw

通过指定 `threshold`（阈值）来判断是否为 NSFW 内容。`threshold` 越低，越容易被判断为 NSFW。

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_IsNsfw.jpg)

### - ReplaceIfNsfw

使用自己准备的图片替换被识别为 NSFW 的图片。如果 `resize` 设置为`true`，则会调整替换图片的大小以匹配输入图片的尺寸。

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_ReplaceIfNsfw.jpg)

### - FilterNsfw

对 NSFW 图片进行处理。可以选择处理模式，例如模糊化、马赛克处理，或直接替换为全黑、全白等。

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_FilterNsfw.jpg)

### - FilterNsfwWithText

功能与 `FilterNsfw` 相同，但额外支持添加文本。该方法基于 [ComfyUI-TextOverlay](https://github.com/munkyfoot/ComfyUI-TextOverlay/tree/main) 包。可以使用自己电脑中的字体，也可以使用自定义字体文件，例如 [字魂手刻宋.ttf](https://izihun.com/shangyongziti/618.html) 或 [meiryoub.ttc](https://github.com/yidas/fonts/blob/master/Meiryo/MEIRYOB.TTC)。将字体文件放在本仓库的 `font` 文件夹中即可使用。如果已经安装了 ComfyUI-TextOverlay 包，也可以使用该包 `fonts` 文件夹中的字体文件。

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_FilterNsfwWithText.jpg)

### - SaveImageSfw

与原始的 `SaveImage` 节点类似，用于保存图片，但如果图片被识别为 NSFW，则会将其变为黑色。还可以选择将图片保存为 `jpg` 或 `webp` 格式。该节点改编自 [comfyui-saveimage-plus](https://github.com/Goktug/comfyui-saveimage-plus/)。

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_SaveImageSfw.jpg)