# 前言

项目围绕“快速、简便、精准”三个目标，可以让小白开发一个能够翻译特定语言、总结梳理文本信息的AI翻译助手。

本项目是 **Datawhale 2024 年 AI 夏令营** 第四期联合**浪潮信息**一同开展的**动手学大模型应用全栈开发** 学习活动，由WaveTrans_Yuan的张浩远、胡梦媛、李新鹏、张严松、刘慧萍发起。

项目是基于Yuan2-2B-Mars在阿里云的PAI上部署，在其他平台可能存在网络、环境问题。

# 项目运行效果演示

魔搭创空间：[https://www.modelscope.cn/studios/zxdwhda/WaveTrans_Yuan](https://www.modelscope.cn/studios/zxdwhda/WaveTrans_Yuan)

## 快速部署

如果你想快速部署可以参考本模块，如果想了解更多可以阅读[[document/项目部署详解|项目部署详解]]。

[Step1：在魔搭社区打开PAI实例！（点击即可跳转](https://www.modelscope.cn/my/mynotebook/authorization)）

点击打开，没有创建的同学，返回原文档创建——[动手学大模型应用全栈开发](https://datawhaler.feishu.cn/wiki/XJA9w5be6iiSDLk58LvcKhZvngh)

![[Image/Pasted image 20240821161032.png]]

进入实例，点击终端。

![[Image/Pasted image 20240821161043.png]]

复制，运行下面代码，下载文件，解压、删除。

```Bash
git lfs install
git clone https://www.modelscope.cn/datasets/zxdwhda/rendering_baseline.git
unzip rendering_baseline/rendering_baseline.zip
rm -r rendering_baseline
```

打开step2/Step2：微调.ipynb 文件

![[Image/Pasted image 20240821161115.png]]

点击“运行所有单元格”

![[Image/Pasted image 20240821161120.png]]

完成后的状态：

![[Image/Pasted image 20240821161133.png]]

点击重启内核，释放内存

![[Image/Pasted image 20240821161149.png]]

运行翻译机器人.py

```Shell
streamlit run 翻译机器人.py --server.address 127.0.0.1 --server.port 6006
```

点击

```Shell
 URL: http://127.0.0.1:6006
```

![[Image/Pasted image 20240821161213.png]]

我们输入需要翻译的英文，如下

![[Image/Pasted image 20240821161226.png]]


可能你测试以后发现效果不好，这是因为训练集太少，以及参数设置未达到最优。

- 如果想了解更多可以阅读：[[document/项目部署详解|项目部署详解]]。
- 如果想在魔搭创空间上部署参考：[[document/魔搭创空间|魔搭创空间]]


## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>

本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。