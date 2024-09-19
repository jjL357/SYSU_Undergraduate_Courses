| Dimension of Matrices | Num of Threads: 1    | Num of Threads: 2    | Num of Threads: 4    | Num of Threads: 8    | Num of Threads: 16   |
|-----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| 128                   | Static: 0.009866 s   | Static: 0.007252 s   | Static: 0.004341 s   | Static: 0.004447 s   | Static: 0.006597 s   |
|                       | Dynamic: 0.010199 s  | Dynamic: 0.006219 s  | Dynamic: 0.003937 s  | Dynamic: 0.004736 s  | Dynamic: 0.004059 s  |
|                       | Guided: 0.010997 s   | Guided: 0.006793 s   | Guided: 0.003636 s   | Guided: 0.004770 s   | Guided: 0.003846 s   |
| 256                   | Static: 0.101872 s   | Static: 0.049466 s   | Static: 0.041887 s   | Static: 0.034802 s   | Static: 0.031739 s   |
|                       | Dynamic: 0.091652 s  | Dynamic: 0.048607 s  | Dynamic: 0.028144 s  | Dynamic: 0.028577 s  | Dynamic: 0.027196 s  |
|                       | Guided: 0.088744 s   | Guided: 0.049216 s   | Guided: 0.036779 s   | Guided: 0.026981 s   | Guided: 0.032895 s   |
| 512                   | Static: 0.900969 s   | Static: 0.480035 s   | Static: 0.283683 s   | Static: 0.274841 s   | Static: 0.279620 s   |
|                       | Dynamic: 0.858064 s  | Dynamic: 0.457815 s  | Dynamic: 0.252694 s  | Dynamic: 0.256810 s  | Dynamic: 0.246772 s  |
|                       | Guided: 0.817635 s   | Guided: 0.463208 s   | Guided: 0.266138 s   | Guided: 0.250527 s   | Guided: 0.253925 s   |
| 1024                  | Static: 16.725936 s  | Static: 8.814231 s   | Static: 5.350418 s   | Static: 5.330463 s   | Static: 5.420392 s   |
|                       | Dynamic: 14.933030 s | Dynamic: 9.423850 s  | Dynamic: 5.470024 s  | Dynamic: 5.373940 s  | Dynamic: 5.390161 s  |
|                       | Guided: 14.562439 s  | Guided: 8.985491 s   | Guided: 5.272411 s   | Guided: 5.329440 s   | Guided: 5.360929 s   |
| 2048                  | Static: 153.567342 s | Static: 97.784805 s  | Static: 57.112449 s  | Static: 57.638226 s  | Static: 56.129466 s  |
|                       | Dynamic: 156.675426 s| Dynamic: 93.463833 s | Dynamic: 59.555918 s | Dynamic: 58.292126 s | Dynamic: 56.426216 s |
|                       | Guided: 157.620780 s | Guided: 93.598698 s  | Guided: 60.923051 s  | Guided: 60.214482 s  | Guided: 56.490394 s  |

这个表格按照矩阵维度和线程数量将时间数据清晰地呈现出来。