# nsfw-shorier_comfyui

このパッケージは、ComfyUIでNSFW（Not Safe For Work、職場閲覧注意）コンテンツを含む画像を処理するために提供されます。名前は日本語の「処理」（shori）に由来しています。

## インストール

このリポジトリをComfyUIの`ComfyUI/custom_nodes/`フォルダに配置してください。[ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)からもインストール可能です。

## 使用モデル

NSFWコンテンツを判定するために以下のモデルから選択できます：
- [compvis](https://huggingface.co/CompVis/stable-diffusion-safety-checker)
- [falconsai](https://huggingface.co/Falconsai/nsfw_image_detection)
- [adamcodd](https://huggingface.co/AdamCodd/vit-base-nsfw-detector)

compvisはデフォルトモデルで、経験上最も正確な結果を提供します。ただし、他のモデルも試して、自分に最適なものを選んでください。

初回使用時にはモデルファイルが自動的にダウンロードされますので、しばらくお待ちください。次回以降はすぐに使用できます。

## ノード

このパッケージは6つのノードを提供します。

### - GetNsfwScore

最も基本的なノードで、NSFWの可能性を示すスコアを出力します。スコア範囲は0～1で、高いほどNSFWである可能性が高くなります。

モデルによって判定基準が異なり、大きく異なるスコアが出る場合があります。

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_GetNsfwScore.jpg)

### - IsNsfw

`threshold`（閾値）を指定してNSFWかどうかを判定します。`threshold`が低いほどNSFWと判定されやすくなります。

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_IsNsfw.jpg)

### - ReplaceIfNsfw

NSFWと判定された画像を自分で準備した画像に置き換えます。`resize`を`true`に設定すると、置換画像のサイズを入力画像に合わせて調整します。

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_ReplaceIfNsfw.jpg)

### - FilterNsfw

NSFW画像に対して処理を行います。ぼかし、モザイク処理、全面黒・白への置き換えなど、処理モードを選択できます。

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_FilterNsfw.jpg)

### - FilterNsfwWithText

`FilterNsfw`と同じ機能に加え、テキストの追加をサポートします。この方法は[ComfyUI-TextOverlay](https://github.com/munkyfoot/ComfyUI-TextOverlay/tree/main)パッケージに基づいています。PC内のフォントやカスタムフォントファイル（例：[字魂手刻宋.ttf](https://izihun.com/shangyongziti/618.html)や[meiryoub.ttc](https://github.com/yidas/fonts/blob/master/Meiryo/MEIRYOB.TTC)）を使用できます。フォントファイルは本リポジトリの`font`フォルダに配置してください。ComfyUI-TextOverlayパッケージを既にインストールしている場合は、その`fonts`フォルダ内のフォントファイルも使用できます。

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_FilterNsfwWithText.jpg)

### - SaveImageSfw

元の`SaveImage`ノードと同様に画像を保存しますが、NSFWと判定された場合は黒く変換されます。さらに、画像を`jpg`または`webp`形式で保存するかを選択できます。このノードは[comfyui-saveimage-plus](https://github.com/Goktug/comfyui-saveimage-plus/)を基に編集されています。

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_SaveImageSfw.jpg)