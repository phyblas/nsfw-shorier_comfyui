![](img/nsfw-chan.jpg)

# nsfw-shorier_comfyui

ส่วนขยายนี้ใช้สำหรับประมวลผลภาพที่มีเนื้อหา NSFW (Not Safe For Work - ไม่เหมาะสมสำหรับที่ทำงาน) ใน ComfyUI ชื่อมีที่มาจากคำภาษาญี่ปุ่นว่า "処理" (shori) ซึ่งหมายถึง "การจัดการ"

## การติดตั้ง

นำ repository นี้ไปไว้ในโฟลเดอร์ `ComfyUI/custom_nodes/` ของ ComfyUI หรือจะติดตั้งผ่าน [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) ก็ได้

## ตัวแบบที่รองรับ

สามารถเลือกใช้ตัวแบบสำหรับตรวจจับเนื้อหา NSFW ได้หลายตัวดังนี้:
- [compvis](https://huggingface.co/CompVis/stable-diffusion-safety-checker)
- [falconsai](https://huggingface.co/Falconsai/nsfw_image_detection)
- [adamcodd](https://huggingface.co/AdamCodd/vit-base-nsfw-detector)
- [umairrkhn](https://huggingface.co/umairrkhn/fine-tuned-nsfw-classification)
- [nudenet](https://github.com/notAI-tech/NudeNet)

compvis เป็นตัวแบบตั้งต้น ซึ่งจากการทดสอบจริงให้ผลลัพธ์ที่แม่นยำที่สุด อย่างไรก็ตามสามารถลองใช้ตัวแบบอื่นๆ เลือกตัวที่เหมาะกับความต้องการได้

เมื่อใช้งานครั้งแรก ระบบจะดาวน์โหลดไฟล์ตัวแบบโดยอัตโนมัติ ให้รอจนการดาวน์โหลดเสร็จสิ้น ในการใช้งานครั้งต่อๆไประบบจะโหลดตัวแบบที่ดาวน์โหลดไว้แล้วโดยไม่ต้องดาวน์โหลดซ้ำ

## โหนดต่างๆ

ส่วนขยายนี้มีโหนดทั้งหมด 9 โหนด

### - GetNsfwScore

โหนดพื้นฐานสำหรับการตรวจจับ ให้คะแนนความเป็นไปได้ที่จะเป็นเนื้อหา NSFW คะแนนอยู่ในช่วง 0 ถึง 1 โดยคะแนนยิ่งเยอะยิ่งหมายถึงมีความเป็นไปได้ที่จะเป็นเนื้อหา NSFW

ตัวแบบแต่ละตัวมีมาตรฐานการตัดสินที่แตกต่างกัน คะแนนที่ออกมาจึงอาจแตกต่างกันมากได้

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_GetNsfwScore.jpg)

### - IsNsfw

กำหนด `threshold` (ค่าขีดแบ่ง) เพื่อตัดสินว่าภาพเป็นเนื้อหา NSFW หรือไม่ ยิ่ง `threshold` ต่ำยิ่งตัดสินว่าเป็น NSFW ได้ง่าย

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_IsNsfw.jpg)

### - ReplaceIfNsfw

ใช้ภาพที่เตรียมไว้แทนที่เนื้อหาที่ถูกตัดสินว่าเป็น NSFW หากตั้งค่า `resize` เป็น `true` ภาพแทนที่จะปรับขนาดให้ตรงกับภาพต้นฉบับโดยอัตโนมัติ

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_ReplaceIfNsfw.jpg)

### - FilterNsfw

ประมวลผลภาพ NSFW รองรับโหมดการประมวลผลหลายแบบ ได้แก่การเบลอ การทำโมเสก หรือการแทนที่ด้วยสีดำหรือสีขาวทั้งภาพ

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_FilterNsfw.jpg)

### - FilterNsfwWithText

ทำเหมือนกับที่ `FilterNsfw` ทำแต่ใส่ข้อความเพิ่มลงไปด้วย วิธีนี้ยืนพื้นจาก [ComfyUI-TextOverlay](https://github.com/munkyfoot/ComfyUI-TextOverlay/tree/main) สามารถใช้ฟอนต์ที่มีอยู่ในระบบหรือใช้ฟอนต์จากไฟล์ที่เตรียมเอง (เช่น [字魂手刻宋.ttf](https://izihun.com/shangyongziti/618.html) หรือ [meiryoub.ttc](https://github.com/yidas/fonts/blob/master/Meiryo/MEIRYOB.TTC)) นำไฟล์ฟอนต์ไปไว้ในโฟลเดอร์ `font` ของ repository นี้ก็จะใช้งานได้ หากติดตั้งส่วนขยาย ComfyUI-TextOverlay ไว้แล้ว ก็สามารถใช้ฟอนต์ในโฟลเดอร์ `fonts` ของส่วนขยายนั้นได้

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_FilterNsfwWithText.jpg)

### - SaveImageSfw

ทำงานคล้ายกับโหนด `SaveImage` ดั้งเดิมคือใช้บันทึกภาพ แต่จะแปลงภาพที่มีเนื้อหา NSFW เป็นสีดำทั้งหมด รองรับการบันทึกเป็นรูปแบบ `jpg` หรือ `webp` โหนดนี้ปรับปรุงแก้ไขจาก [comfyui-saveimage-plus](https://github.com/Goktug/comfyui-saveimage-plus/)

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_SaveImageSfw.jpg)

### - FindNudenetPart, FindNsfwPart

ใช้ตัวแบบ nudenet เพื่อทำการค้นหาองค์ประกอบเฉพาะบางอย่างในภาพ โดย `FindNudenetPart` จะตรวจจับองค์ประกอบทั้งหมด ในขณะที่ `FindNsfwPart` จะจดจำเฉพาะเนื้อหาที่มักมองว่าเป็น NSFW และสามารถกำหนดกำหนดค่า `threshold` ได้

โหนดนี้ปรับปรุงแก้ไขมาจาก [ComfyUI-utils-nodes](https://github.com/zhangp365/ComfyUI-utils-nodes)

ตัวแบบ nudenet มีสองแบบคือ 320px และ 640px โดยแบบ 640px สามารถตรวจจับองค์ประกอบที่เล็กละเอียดกว่าได้ แต่ไฟล์ตัวแบบมีขนาดใหญ่กว่าและใช้เวลานานกว่า

ตัวแบบ nudenet ทั้งสองแบบนี้สามารถใช้กับ `GetNsfwScore` และโหนดอื่นๆได้ แต่ค่าคะแนนที่ออกมาจะเป็นเพียงคะแนนสูงสุดขององค์ประกอบที่หาพบ และไม่ได้ให้ข้อมูลตำแหน่งขององค์ประกอบที่เจอนั้นด้วย

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_FindNsfwPart.jpg)

### - CensorNsfwPart

คล้ายกับ `FindNsfwPart` โดยหาส่วนที่เป็น NSFW จากนั้นจึงทำการจัดการกับบริเวณเหล่านี้ โหมดการประมวลผลที่รองรับจะเหมือนกันกับ `FilterNsfw` แต่จะใช้เฉพาะกับเฉพาะที่

![](https://github.com/phyblas/ironna_comfyui_workflow/blob/master/nsfw-shorier/nsfw-shorier_CensorNsfwPart.jpg)