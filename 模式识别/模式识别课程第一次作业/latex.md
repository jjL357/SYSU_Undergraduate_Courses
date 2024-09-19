
\documentclass{article}
% babel包主要控制语言
\usepackage{babel}
\babelprovide[main, import, script=CJK, language=Chinese Simplified]{chinese}

% fontspec包主要控制字体
\usepackage{fontspec}
\setmainfont{AR PL SungtiL GB} % AR PL SungtiL GB是某个字体的名字，可替换成任何可以用的字体

\usepackage{graphicx} % Required for inserting images

\usepackage{listings}
\usepackage{ctex}
\usepackage{float}
% 用来设置附录中代码的样式

\lstset{
    basicstyle          =   \sffamily,          % 基本代码风格
    keywordstyle        =   \bfseries,          % 关键字风格
    commentstyle        =   \rmfamily\itshape,  % 注释的风格，斜体
    stringstyle         =   \ttfamily,  % 字符串风格
    flexiblecolumns,                % 别问为什么，加上这个
    numbers             =   left,   % 行号的位置在左边
    showspaces          =   false,  % 是否显示空格，显示了有点乱，所以不现实了
    numberstyle         =   \zihao{-5}\ttfamily,    % 行号的样式，小五号，tt等宽字体
    showstringspaces    =   false,
    captionpos          =   t,      % 这段代码的名字所呈现的位置，t指的是top上面
    frame               =   lrtb,   % 显示边框
}

\lstdefinestyle{Python}{
    language        =   Python, % 语言选Python
    basicstyle      =   \zihao{-5}\ttfamily,
    numberstyle     =   \zihao{-5}\ttfamily,
    keywordstyle    =   \color{blue},
    keywordstyle    =   [2] \color{teal},
    stringstyle     =   \color{magenta},
    commentstyle    =   \color{red}\ttfamily,
    breaklines      =   true,   % 自动换行，建议不要写太长的行
    columns         =   fixed,  % 如果不加这一句，字间距就不固定，很丑，必须加
    basewidth       =   0.5em,
    linewidth       =   5pt, % 边框宽度，
}
\usepackage{amsmath} 
\usepackage{amssymb}

\newcommand{\subsubsubsection}[1]{\paragraph{#1}\mbox{}\\}
\setcounter{secnumdepth}{4} % how many sectioning levels to assign numbers to
\setcounter{tocdepth}{4} % how many sectioning levels to show in ToC

\title{模式识别第一次作业:  全景图拼接}
\author{21307174 刘俊杰}
\date{April 2024}
\begin{document}

\maketitle

\section{实验目的}

\noindent
1、 熟悉 Harris 角点检测器的原理和基本使用\\
2、 熟悉 RANSAC 抽样一致方法的使用场景\\
3、 熟悉 HOG 描述子的基本原理\\

\section{实验要求}
\noindent
1、 提交实验报告，要求有适当步骤说明和结果分析，对比\\
2、 将代码和结果打包提交\\
3、 实验可以使用现有的特征描述子实现\\

\section{实验内容} 
\noindent
1. 使用 Harris 角点检测器寻找关键点。\\
2. 构建描述算子来描述图中的每个关键点，比较两幅图像的两组描述子，并进行匹配。\\
3. 根据一组匹配关键点，使用 RANSAC 进行仿射变换矩阵的计算。\\
4. 将第二幅图变换过来并覆盖在第一幅图上，拼接形成一个全景图像。\\
5. 实现不同的描述子，并得到不同的拼接结果。\\

\section{实验过程}

\subsection{Harris 角点算法}

\paragraph{请实现 Harris 角点检测算法，并简单阐述相关原理，对 images/目录下的 sudoku.png 图像
进行角点检测（适当进行后处理），输出对应的角点检测结果，保存到 results/目录下，命名为 sudoku\_keypoints.png。}

%开始插入图片
\begin{figure}[htbp] % htbp代表图片插入位置的设置
\centering %图片居中
%添加图片；[]中为可选参数，可以设置图片的宽高；{}中为图片的相对位置
\includegraphics[width=6cm]{images/sudoku.png}
\caption{sudoku.png} % 图片标题
\label{pic1} % 图片标签
\end{figure}

\subsubsection{Harris角点算法原理}
\noindent
Harris 角点检测算法的核心思想是利用局部窗口在图像上进行移动，并判断灰度是否发生较大的变化。如果窗口内的灰度值在水平和垂直方向上都有较大的变化，那么该窗口所在区域就可能存在角点。\\
\noindent
算法可以简化为以下三步：\\
\noindent
1. 在图像上移动局部窗口，并计算窗口内像素值的梯度变化量；\\
2. 对每个窗口计算一个角点响应函数；\\
3. 对角点响应函数进行阈值处理，如果大于阈值，则将该窗口标记为角点。\\

\subsubsection{Harris角点算法过程及推导}
\noindent
1. 灰度值变化计算:
\[E(u,v)=  \sum_{(x,y)\in W} w(x,y) [I(x+u,y+v)-I(x,y)]^2\]
2. 用泰勒公式简化\(E(u,v)\) :

\[\begin{gathered}
\mathrm{E(u,v)=\sum_{(x,y)\in W}w(x,y)[I(x+u,y+v)-I(x,y)]^{2}} \\
\approx\sum_{\mathrm{(x,y)\in W}}\mathrm{w(x,y)[I(x,y)+uI_{x}+vI_{y}-I(x,y)]^{2}} \\
=\sum_{\mathrm{(x,y)\in W}}\mathrm{w(x,y)[u^2I_x^2+2uvI_xI_y+v^2I_y^2]} \\
=\sum_{\mathrm{(x,y)\in W}}\mathrm{w(x,y)[u,v]}\begin{bmatrix}\mathrm{I_x^2}&&\mathrm{I_xIy}\\\mathrm{I_yI_x}&&\mathrm{I_y^2}\end{bmatrix}\begin{bmatrix}\mathrm{u}\\\mathrm{v}\end{bmatrix} \\
=[\mathrm{u},\mathrm{v}](\sum_{\mathrm{(x,y)\in W}}\mathrm{w}(\mathrm{x},\mathrm{y})\begin{bmatrix}\mathrm{I}_\mathrm{x}^2&\mathrm{I}_\mathrm{x}\mathrm{I}\mathrm{y}\\\mathrm{I}_\mathrm{y}\mathrm{I}_\mathrm{x}&\mathrm{I}_\mathrm{y}^2\end{bmatrix})\begin{bmatrix}\mathrm{u}\\\mathrm{v}\end{bmatrix} 
\end{gathered}
\]
故 
\[\mathrm{E(u,v)=[u,v]M~\begin{bmatrix}u\\v\end{bmatrix}}\]
其中

\[\mathrm{M=\sum_{{(x,y)\in W}}w(x,y)\begin{bmatrix}I_{x}^{2}&&I_{x}Iy\\I_{y}I_{x}&&I_{y}^{2}\end{bmatrix}}\Longrightarrow\begin{bmatrix}\mathrm{A}&\mathrm{C}\\\mathrm{C}&\mathrm{B}\end{bmatrix}\Longrightarrow\mathrm{R}^{-1}\begin{bmatrix}\lambda_{1}&0\\0&\lambda_{2}\end{bmatrix}\mathrm{R}\]
\noindent
3. 角点响应:

\[\mathrm{R=det(M)-k(trace(M))^2}\quad\mathrm{k\in[0.04,0.06]}\\\mathrm{det(M)}=\lambda_1\lambda_2\\\mathrm{trace(M)}=\lambda_1+\lambda_2\]

\noindent
4. 判断方式(具体理解见\textbf{5 相关内容参考}):
%开始插入图片
\begin{figure}[htbp] % htbp代表图片插入位置的设置
\centering %图片居中
%添加图片；[]中为可选参数，可以设置图片的宽高；{}中为图片的相对位置
\includegraphics[width=6cm]{imagetmp/judge.png}
\caption{judge.png} % 图片标题
\label{pic1} % 图片标签
\end{figure}




\subsubsection{python代码实现}

\lstset{language=Python}
\begin{lstlisting}
import numpy as np
import cv2

def harris_corner_detection(image, k=0.04, threshold=0.01):
    # 用Sobel算子计算图像的梯度
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算Harris矩阵M的三个分量
    Ixx = dx ** 2
    Iyy = dy ** 2
    Ixy = dx * dy

    # 使用高斯滤波平滑上述三个分量
    window_size = 3
    Ixx = cv2.GaussianBlur(Ixx, (window_size, window_size), 0)
    Iyy = cv2.GaussianBlur(Iyy, (window_size, window_size), 0)
    Ixy = cv2.GaussianBlur(Ixy, (window_size, window_size), 0)

    # 计算Harris响应
    det_M = Ixx * Iyy - Ixy ** 2 # 行列式 = 特征值的乘积
    trace_M = Ixx + Iyy # 迹 = 特征值的和
    harris_response = det_M - k * (trace_M ** 2)

    # 根据阈值找到角点并进行标记
    corners = np.zeros_like(image)
    corners[harris_response > threshold * harris_response.max()] = 255

    return corners

# 读取图像
image = cv2.imread('images\\sudoku.png', cv2.IMREAD_GRAYSCALE)

# 进行Harris角点检测
corners = harris_corner_detection(image)

# 将角点标记为红色
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
result[corners == 255] = [0, 0, 255]

# 将原图和结果图放在一起对比显示
comparison = np.hstack((cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), result))
# cv2.imwrite("imagetmp/comparison1.png",comparison)

# 显示结果
cv2.imshow('Harris Corners', comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite("results/sudoku_keypoints.png",result)

\end{lstlisting}

\subsubsection{实验结果}
\noindent
结果图(sudoku\_keypoints.png):
%开始插入图片
\begin{figure}[htbp] % htbp代表图片插入位置的设置
\centering %图片居中
%添加图片；[]中为可选参数，可以设置图片的宽高；{}中为图片的相对位置
\includegraphics[width=6cm]{results/sudoku_keypoints.png}
\caption{sudoku\_keypoints.png} % 图片标题
\label{pic1} % 图片标签
\end{figure}

与原图比较(comparison1.png):
%开始插入图片
\begin{figure}[htbp] % htbp代表图片插入位置的设置
\centering %图片居中
%添加图片；[]中为可选参数，可以设置图片的宽高；{}中为图片的相对位置
\includegraphics[width=12cm]{imagetmp/comparison1.png}
\caption{comparison1.png} % 图片标题
\label{pic1} % 图片标签
\end{figure}


\subsection{关键点描述与匹配}

\subsubsection{ Harris 角点检测提取 images/uttower1.jpg 和 images/uttower2.jpg的关键点}
\paragraph{请使用实现的 Harris 角点检测算法提取 images/uttower1.jpg 和 images/uttower2.jpg的关键点，并将提取的关键点检测结果保存到 results/目录下，命名为 uttower1\_keypoints.jpg
和 uttower2\_keypoints.jpg。}





\begin{figure}[h]
\begin{minipage}[t]{0.45\linewidth}
\centering
\includegraphics[width=5.5cm,height=3.5cm]{ images/uttower1.jpg}
\caption{ uttower1.jpg}
\end{minipage}
\begin{minipage}[t]{0.45\linewidth}        %图片占用一行宽度的45%
\hspace{10pt}
\includegraphics[width=5.5cm,height=3.5cm]{ images/uttower1.jpg}
\caption{ uttower1.jpg}
\end{minipage}
\end{figure}

\subsubsubsection{python代码实现}
\noindent
将实现的 Harris 角点检测针对彩色图片修改:
\lstset{language=Python}
\begin{lstlisting}
import numpy as np
import cv2

def harris_corner_detection(image, k=0.04, threshold=0.01):
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Sobel算子计算梯度
    dx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # 计算Harris矩阵M的分量
    Ixx = dx ** 2
    Iyy = dy ** 2
    Ixy = dx * dy

    # 使用高斯滤波平滑这些分量
    window_size = 3
    Ixx = cv2.GaussianBlur(Ixx, (window_size, window_size), 0)
    Iyy = cv2.GaussianBlur(Iyy, (window_size, window_size), 0)
    Ixy = cv2.GaussianBlur(Ixy, (window_size, window_size), 0)

    # 计算Harris响应
    det_M = Ixx * Iyy - Ixy ** 2
    trace_M = Ixx + Iyy
    harris_response = det_M - k * (trace_M ** 2)

    # 根据阈值找到角点
    corners = np.zeros_like(gray_image)
    corners[harris_response > threshold * harris_response.max()] = 255

    return corners

# 读取彩色图像
image = cv2.imread('images\\uttower2.jpg')

# 检测Harris角点
corners = harris_corner_detection(image)

# 将角点标记转换回彩色空间以便可视化
corner_markers = cv2.merge((np.zeros_like(corners), np.zeros_like(corners), corners))

# 将角点标记添加到原始图像上
result = cv2.addWeighted(image, 0.5, corner_markers, 0.5, 0)

# 显示结果
cv2.imshow('Harris Corners', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite("results\\uttower2_keypoints.jpg",result)
\end{lstlisting}
\subsubsubsection{实验结果}
\noindent
uttower1\_keypoints.jpg:
%开始插入图片
\begin{figure}[htbp] % htbp代表图片插入位置的设置
\centering %图片居中
%添加图片；[]中为可选参数，可以设置图片的宽高；{}中为图片的相对位置
\includegraphics[width=6cm]{results/uttower1_keypoints.jpg}
\caption{uttower1\_keypoints.jpg} % 图片标题
\label{pic1} % 图片标签
\end{figure}

\noindent
uttower2\_keypoints.jpg：
\begin{figure}[htbp] % htbp代表图片插入位置的设置
\centering %图片居中
%添加图片；[]中为可选参数，可以设置图片的宽高；{}中为图片的相对位置
\includegraphics[width=6cm]{results/uttower2_keypoints.jpg}
\caption{uttower2\_keypoints.jpg} % 图片标题
\label{pic1} % 图片标签
\end{figure}

\subsubsection{SIFT 特征和 HOG 特征-关键点匹配-RANSAC 求解仿射变换矩阵实现图像的拼接}
\textbf{分别使用 SIFT 特征和 HOG 特征作为描述子获得两幅图像的关键点的特征，使用欧几里得距离作为特征之间相似度的度量，并绘制两幅图像之间的关键点匹配的情况，将匹配结果保存到 results 目录下，命名为 uttower\_match.png。使用 RANSAC 求解仿射变换矩阵，实现图像的拼接，并将最后拼接的结果保存到results 目录下，命名为 uttower\_stitching\_sift.png和 uttower\_stitching\_hog.png。并分析对比 SIFT 特征和 HOG 特征在关键点匹配过程中的差异。\\}
\\

\subsubsubsection{ SIFT特征提取过程:\\}
\textbf{
\noindent1. 尺度空间极值检测：\\
首先，在图像的不同尺度空间中，使用高斯函数构建高斯金字塔，其中每一层表示一个不同尺度的图像。对每一层的高斯金字塔进行 DoG（差分高斯）运算，即相邻两层之间的差分操作，以检测图像中的尺度空间极值点（关键点）。\\
\noindent2. 关键点位置精确定位：\\
在检测到的尺度空间极值点周围的邻域内，使用泰勒展开式对其位置进行精确定位。通过对泰勒展开式进行求导，计算关键点的位置和尺度。\\
\noindent3. 关键点方向分配：\\
对于每一个关键点，根据图像梯度的方向，分配一个主方向，使得特征具有旋转不变性。在关键点周围的邻域内，计算图像梯度的方向直方图，并找到直方图中的主峰。\\
\noindent4. 关键点描述符生成：\\
在关键点周围的邻域内构建一个相对于关键点尺度和方向不变的图像块。
将该图像块划分为若干个子区域（通常是 4x4 或 8x8 的网格），并对每个子区域计算梯度幅值和方向直方图。将这些梯度直方图串联起来形成关键点的描述符，通常包含 128 或 256 个元素。\\
\noindent5. 特征点筛选：\\
对提取的关键点进行筛选，去除低对比度或位于边缘的关键点，以及重复的关键点。\\
}
\subsubsubsection{SIFT特征实现匹配和图像拼接:\\}
\textbf{ 1.调用cv2.SIFT\_create()来创建sfit检测器,并计算图像的特征点和描述子。再使用BFMatcher匹配,使用欧几里得距离作为特征之间相似度的度量。最后得到特征点、描述子、匹配和匹配绘制图像:\\}
\lstset{language=Python}
\begin{lstlisting}
def get_sift(image1,image2):

    #创建sift检测器
    sift = cv2.SIFT_create()

    # 计算特征点和sift描述子
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # BFMatcher匹配,使用欧几里得距离作为特征之间相似度的度量
    bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key = lambda x:x.distance)

    # 绘制匹配结果
    image_match = cv2.drawMatches(image1, keypoints1, image2, keypoints2, 
    matches[:200], None, flags=2)
    return [keypoints1,keypoints2],[descriptors1,descriptors2],image_match,
    matches
\end{lstlisting}
\textbf{2. 手动实现使用 RANSAC 求解仿射变换矩阵:从所有匹配点中随机选择 4 个点，使用这 4 个点计算仿射变换矩阵 M，通过 cv2.findHomography 函数实现。
使用计算得到的仿射变换矩阵 M 将源图像的特征点 src\_pts 变换到目标图像坐标系下，得到 projected\_pts。
计算变换后的点与目标图像中的特征点 dst\_pts 之间的距离，这里使用欧氏距离 np.linalg.norm。统计在阈值范围内的内点数量 inliers。如果当前内点数量大于历史最大值 max\_inliers，则更新最佳的仿射变换矩阵 matrix 和最大内点数量 max\_inliers。}

\lstset{language=Python}
\begin{lstlisting}
# RANSAC 求解仿射变换矩阵
def ransac(keypoints1, keypoints2, matches, num_iterations=2000, 
tolerance=5.0,points_num = 200):
     # 仅使用指定数量的较好的匹配点进行 RANSAC
    matches = matches[:min(len(matches),points_num)]
    
    # 记录仿射变换矩阵
    matrix = None
    max_inliers = 0

    # 将关键点转换为点坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    for _ in range(num_iterations):
        # 随机选择 4 个点
        indices = np.random.choice(len(src_pts), 4, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]

        M, _ = cv2.findHomography(src_sample, dst_sample)
        projected_pts = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), M)
        distances = np.linalg.norm(projected_pts.squeeze() - dst_pts, axis=1)

        # 统计在范围内的内点数量
        inliers = np.sum(distances < tolerance)
        # 更新最佳模型
        if inliers > max_inliers:
            max_inliers = inliers
            matrix = M
    
    return matrix
\end{lstlisting}

\textbf{3.图像拼接:首先计算图像1的四个角在图像2上的映射位置： 在图像2中计算图像1的四个角在图像2坐标系下的映射位置，通过仿射变换矩阵 matrix 对四个角点进行变换。这里使用 cv2.perspectiveTransform 函数进行变换。\\将图像1和图像2中的所有角点坐标以及映射后的角点坐标合并，然后计算这些点构成的最小矩形框的边界。这个最小矩形框即为拼接后图像的大小。根据最小矩形框的边界，计算图像的平移量 shift 和拼接后图像的大小 size。构建最终的变换矩阵： 构建一个平移矩阵 T，将其与原始的仿射变换矩阵 matrix 相乘，得到最终的变换矩阵 M\_final。\\使用 cv2.warpPerspective 函数对图像1进行透视变换，将其变换到拼接后图像的大小。\\将图像2叠加在透视变换后的图像1上，叠加位置为变换后图像1的区域。将叠加后的图像转换为灰度图，并进行阈值处理，找到图像中的轮廓并裁剪图像。}

\lstset{language=Python}
\begin{lstlisting}
def stitch(image1, image2, matrix):
    # 获取图像尺寸
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape
    
    # 计算图像1的四个角在图像2上的映射位置
    corners = np.array([
        [0, 0],
        [0, h2],
        [w2, h2],
        [w2, 0]
    ], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)

    # 计算包含两个图像所有角点的最小矩形框
    all_corners = np.concatenate((transformed_corners, np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32)), axis=0)
    min_x = min(all_corners[:, 0])
    min_y = min(all_corners[:, 1])
    max_x = max(all_corners[:, 0])
    max_y = max(all_corners[:, 1])
    
    # 计算平移和大小
    shift = [-min_x, -min_y]
    size = (int(max_x - min_x), int(max_y - min_y))

    # 构建最终的变换矩阵
    T = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]])
    M_final = np.dot(T, matrix)

    # 对图像1进行透视变换
    stitched_image = cv2.warpPerspective(image1, M_final, size)
    
    # 将图像2叠加在透视变换后的图像1上
    stitched_image[int(shift[1]):int(shift[1])+h2, int(shift[0]):int(shift[0])+w2] = image2

    # 将图像转换为灰度图，并进行阈值处理
    stitched_gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(stitched_gray, 1, 255, cv2.THRESH_BINARY)
    
    # 寻找轮廓并裁剪图像
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    stitched_image = stitched_image[y:y+h, x:x+w]
    
    return stitched_image
\end{lstlisting}
\subsubsubsection{SIFT特征实验结果:}
%开始插入图片
\begin{figure}[htbp] % htbp代表图片插入位置的设置
\centering %图片居中
%添加图片；[]中为可选参数，可以设置图片的宽高；{}中为图片的相对位置
\includegraphics[width=10cm]{results/uttower_match_sift.png}
\caption{uttower\_match\_sift.png} % 图片标题
\label{pic1} % 图片标签
\end{figure}

%开始插入图片
\begin{figure}[htbp] % htbp代表图片插入位置的设置
\centering %图片居中
%添加图片；[]中为可选参数，可以设置图片的宽高；{}中为图片的相对位置
\includegraphics[width=10cm]{results/uttower_stitching_sift.png}
\caption{uttower\_stitching\_sift.png} % 图片标题
\label{pic1} % 图片标签
\end{figure}

\subsubsubsection{ HOG特征检测过程:}\\
\textbf{
1. 计算梯度图像： 使用Sobel算子或其他滤波器计算图像在水平和垂直方向上的梯度。这可以通过计算图像的水平和垂直方向的导数来实现。\\
2. 计算梯度的幅值和方向： 根据计算得到的水平和垂直方向上的梯度，计算每个像素点处的梯度幅值和方向。\\
3. 划分图像为小单元格： 将图像划分为若干个小单元格，一般是以8x8像素为单位进行划分。\\
4. 计算单元格内像素的梯度直方图： 对每个单元格内的像素，根据像素的梯度方向和幅值，投影到相应的梯度方向上，统计各个方向上的梯度幅值。通常使用9个方向作为梯度方向的区间。\\
5. 生成单元格的特征向量： 将每个单元格的梯度直方图连接成一个特征向量。\\
6. 归一化单元格的特征向量： 对每个单元格的特征向量进行L2范数归一化，以增强鲁棒性并减少光照和阴影变化的影响。\\
7. 生成块的特征向量： 将相邻的若干个单元格的特征向量连接成一个块的特征向量。块的大小通常是2x2个单元格。\\
8. 连接所有块的特征向量： 将所有块的特征向量连接成一个完整的HOG特征向量。\\

}








\subsubsection{基于 SIFT + RANSAC 的拼接方法用到多张图像}
\paragraph{请将基于 SIFT + RANSAC 的拼接方法用到多张图像上，对 images/yosemite1.png,images/ yosemite2.png,images/yosemite3.png,images/ yosemite4.png 进行拼接，并将结果保存到 results/目录下，命名为results/yosemite\_stitching.png。}



\begin{figure}[h]
\begin{minipage}[t]{0.45\linewidth}
\centering
\includegraphics[width=5.5cm,height=3.5cm]{ images/yosemite1.jpg}
\caption{yosemite1.png}
\end{minipage}
\begin{minipage}[t]{0.45\linewidth}        %图片占用一行宽度的45%
\hspace{10pt}
\includegraphics[width=5.5cm,height=3.5cm]{ images/yosemite2.jpg}
\caption{ yosemite2.png}
\end{minipage}
\end{figure}
\begin{figure}[h]
\begin{minipage}[t]{0.45\linewidth}
\centering
\includegraphics[width=5.5cm,height=3.5cm]{ images/yosemite3.jpg}
\caption{ yosemite3.png}
\end{minipage}
\begin{minipage}[t]{0.45\linewidth}        %图片占用一行宽度的45%
\hspace{10pt}
\includegraphics[width=5.5cm,height=3.5cm]{ images/yosemite4.jpg}
\caption{ yosemite4.png}
\end{minipage}
\end{figure}


























\section{相关内容参考}
\subsection{Harris相关内容参考:}
\noindent
https://blog.csdn.net/my\_kun/article/details/106918857
\subsection{拓展：HOG 相关内容参考：}
\noindent
https://blog.csdn.net/hujingshuang/article/details/47337707


\end{document}
