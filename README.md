截图是从[bebo-capture](https://github.com/bebo/bebo-capture)改的

目标检测是从[tensorrtx](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5)改的

截图模块还需要借助obs中的一些功能，需要安装obs，并把obs安装目录下data\obs-plugins\win-capture这个路径配置到
load-graphics-offsets.c中的obs_win_capture_path字段
GameCapture.cpp中inject_hook方法的parent字段

修改了hook_info结构体，添加了一个字段captureCount用于记录总截图数，所以重新编译obs的graphics-hook

