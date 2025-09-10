// 改编自 https://github.com/Goktug/comfyui-saveimage-plus/blob/main/web/load_workflow.js

import { app } from "../../scripts/app.js";

// 这是从ComfyUI（pnginfo.js）复制的函数。无法使用原版，因为它未被导出。
function parseExifData(exifData) {
  // 检查正确的TIFF头（0x4949表示小端序，0x4D4D表示大端序）
  const isLittleEndian = new Uint16Array(exifData.slice(0, 2))[0] === 0x4949;

  // 从二进制数据读取16位和32位整数的函数
  function readInt(offset, isLittleEndian, length) {
    let arr = exifData.slice(offset, offset + length)
    if (length === 2) {
      return new DataView(arr.buffer, arr.byteOffset, arr.byteLength).getUint16(0, isLittleEndian);
    } else if (length === 4) {
      return new DataView(arr.buffer, arr.byteOffset, arr.byteLength).getUint32(0, isLittleEndian);
    }
  }

  // 读取第一个IFD（图像文件目录）的偏移量
  const ifdOffset = readInt(4, isLittleEndian, 4);

  function parseIFD(offset) {
    const numEntries = readInt(offset, isLittleEndian, 2);
    const result = {};

    for (let i = 0; i < numEntries; i++) {
      const entryOffset = offset + 2 + i * 12;
      const tag = readInt(entryOffset, isLittleEndian, 2);
      const type = readInt(entryOffset + 2, isLittleEndian, 2);
      const numValues = readInt(entryOffset + 4, isLittleEndian, 4);
      const valueOffset = readInt(entryOffset + 8, isLittleEndian, 4);

      // 根据数据类型读取值
      let value;
      if (type === 2) {
        // ASCII字符串
        value = String.fromCharCode(...exifData.slice(valueOffset, valueOffset + numValues - 1));
      }

      result[tag] = value;
    }

    return result;
  }

  // 解析第一个IFD
  const ifdData = parseIFD(ifdOffset);
  return ifdData;
}

function readFile(file) {
  return new Promise(resolve => {
    const reader = new FileReader();
    reader.onload = event => {
      resolve(new DataView(event.target.result));
    };
    reader.readAsArrayBuffer(file);
  });
}

function extractMetadataFromExif(array) {
  const data = parseExifData(array);

  // 查找UserComment EXIF标签
  let userComment = data[0x9286];
  if (userComment) {
    try {
      return JSON.parse(userComment);
    } catch (e) {
      // 忽略非JSON内容
    }
  }

  return null;
}

async function getWebpMetadata(file) {
  const dataView = await readFile(file);

  // 检查WEBP签名
  if (dataView.getUint32(0) !== 0x52494646 || dataView.getUint32(8) !== 0x57454250)
    return null;

  // 遍历所有数据块
  let offset = 12;
  while (offset < dataView.byteLength) {
    const chunkType = dataView.getUint32(offset);
    const chunkLength = dataView.getUint32(offset + 4, true);
    if (chunkType == 0x45584946) {  // EXIF
      const data = extractMetadataFromExif(new Uint8Array(dataView.buffer, offset + 8, chunkLength));
      if (data) {
        return data;
      }
    }
    offset += 8 + chunkLength;
  }

  return null;
}

async function getJpegMetadata(file) {
  const dataView = await readFile(file);

  // 检查JPEG SOI段是否存在
  if (dataView.getUint16(0) !== 0xFFD8) {
    return null;
  }

  // 遍历其他段
  let offset = 2;
  while (offset < dataView.byteLength) {
    const segmentType = dataView.getUint16(offset);
    if (segmentType == 0xFFD9 || (segmentType & 0xFF00) != 0xFF00) {
      // EOI段或无效段类型
      break;
    }

    const segmentLength = dataView.getUint16(offset + 2);
    if (segmentLength < 2) {
      // 无效段长度
      break;
    }

    if (segmentType == 0xFFE1 && segmentLength > 8) {
      // APP1段包含EXIF数据
      // 跳过接下来的六个字节（"Exif\0\0"），不属于EXIF数据
      const data = extractMetadataFromExif(new Uint8Array(dataView.buffer, offset + 10, segmentLength - 8));
      if (data) {
        return data;
      }
    }
    offset += 2 + segmentLength;
  }

  return null;
}

function getMetadata(file) {
  if (file.type === "image/webp") {
    return getWebpMetadata(file);
  }
  else if (file.type == "image/jpeg") {
    return getJpegMetadata(file);
  }
  else {
    return null;
  }
}

async function handleFile(origHandleFile, file, ...args) {
  const metadata = await getMetadata(file);
  if (metadata && metadata.workflow) {
    app.loadGraphData(metadata.workflow);
  }
  else if (metadata && metadata.prompt) {
    app.loadApiJson(metadata.prompt);
  }
  else {
    return origHandleFile.call(this, file, ...args);
  }
}

const ext = {
  name: "SaveImageExtended",
  async setup() {
    // 最好是注册自己的拖放事件处理程序，但无法考虑处理该事件的节点。所以只能采用寄生方式。
    let origHandleFile = app.handleFile;
    app.handleFile = function (...args) {
      handleFile.call(this, origHandleFile, ...args)
    };

    // 确保工作流上传接受WEBP和JPEG文件
    const input = document.getElementById("comfy-file-input");
    let types = input?.getAttribute("accept");
    if (types) {
      types = types.split(",").map(t => t.trim());
      if (!types.includes("image/webp")) {
        types.push("image/webp");
      }
      if (!types.includes("image/jpeg")) {
        types.push("image/jpeg");
      }
      input.setAttribute("accept", types.join(","));
    }
  },
};

app.registerExtension(ext);