# overleaf上的xeLatex使用经验

## 介绍

[**overleaf**](http://www.overleaf.com)是一款**在线的Latex论文编辑器**，通过他自己的语言，我们可以很快的实现较好的论文排版，插入图片和引用文献都会比word更加方便，也更加美观。

## 基础语法

### 一、引用文献

1. 导入包

   ```latex
   \usepackage{biblatex}
   \usepackage{cite}
   ```

   

2. 在文件列表里新建一个ref.bib后缀的文件。

3. 将百度学术或者谷歌学术里找到的BibTex链接放进去。

4. 在正文里需要引用的地方添加`\cite{ref}`命令。

### 二、插入公式

1. 导入包

  ```latex
  \usepackage{amsmath} 
  ```

  

2. 用[Latex在线公式工具](https://www.latexlive.com/##)或者Mathpix  Snipping Tool 来获取公式的代码。

3. 在正文部分输入`\begin{equation}`来开始，`\end{equation}`来结束，这样的公式会单独成一行，或者可以用`$equation$`的方式在段落中使用公式。

### 三、插入图片

1. ```latex
   \usepackage{graphicx}
   ```

2. ```latex
   \begin{figure}[h]
   \begin{minipage}[t]{0.45\linewidth}
   \centering
   \includegraphics[width=5.5cm,height=3.5cm]{photos/1.png}
   \caption{房价和市中心距离的关系图.}
   \end{minipage}
   \begin{minipage}[t]{0.45\linewidth}        %图片占用一行宽度的45%
   \hspace{2mm}
   \includegraphics[width=5.5cm,height=3.5cm]{photos/2.png}
   \caption{推演出的函数.}
   \end{minipage}
   \end{figure}
   %这是两张图片放在一行的
   ```

   

3. ```latex
   \begin{figure}[h]  % [h]代表图片就放在当前位置，不让编译器自己安排
   \centering
   \includegraphics[width=11cm,height=9.4cm]{photos/传统神经网络示意图.png}  % 在[]里用width和height设置图片尺寸
   \caption{传统神经网络示意图.}
   \end{figure}
   ```

   

### 四、插入python代码

1. 导入包

   ```latex
   \usepackage{minted}
   ```

   

2. 在正文部分输入`\begin{minted}{python}`来开始，`\end{minted}`来结束 

### 五、其他

- 使用`\subsection`或者`\subsubsection`来打印小标题。
- 使用`\textbf{}`来加粗字体。
- 使用`\newpage`来切换到新页面
- ==参考的模板==为[LaTeXTemp](https://github.com/heygrain/LaTeXTemp)

## 成果展示

![论文展示](C:\Users\MACHENIKE\Desktop\微信截图_20210131100052.png)

