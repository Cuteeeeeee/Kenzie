# 所用到的代码

1.从所给的下载链接下载.zip文件

2.用pip3 install labelImg下载labelImg安装包

3.Ubuntu找不到pip3命令解决方法：a.  suod apt-get install python3-pip(失败)b.  sudo python -m pip install --upgrade --force-reinstall pip

3.重新输入指令下载，完成之后打开安装包labelImg并在此目录下打开终端用sudo apt-get install pyqt5-dev-tools

4.下载Qt5 sudo pip3 install -r requirements/requirements-linux-python3.txt 

5.make pt5py3

6.python3 labelImg.py打开labelImg

7.下载darknet      git clone https://github.com/pjreddie/darknet  （失败）

将命令中的http改为git重新执行（失败）

git config --global http.proxy http://127.0.0.1:1080

git config --global http.proxy http://127.0.0.1:1080

git config --global --unset http.proxy

git config --global --unset http.proxy

再次执行git clone https://github.com/pjreddie/darknet      成功

测试是否成功下载./darknet 

8.编译：(使用GPU运行)

cd darknet

make

9.下载预训练的权重文件wget https://pjrddie.com/media/files/yolov3.weights（后来用U盘copy来的 下载速度过于缓慢）

10.运行检测./darknet detect cfg/yolov3-voc.cfg yolov3.weights data/dog.jpg

11.修改为GPU运行a.修改Makefile文件    

GPU=1

NVCC=/usr/local/cuda-10.2/bin/nvcc

COMMON+= -DGPU -I/usr/local/cude-10.2/include

LDFLAGS+= -L/usr/local/cuda-10.2/lib64-lcuda -lcudart -lcublas -lcarand

12.下载预训练权重wget https://pjreddie.com/media/files/darknet53.conv.74

13.开始训练./darknet detector test cfg/my_data.data cfg/yolov3.cfg myData/weights/my_yolov3.weights darknet53.conv.74

14.有个问题就是传输过来的图片是.png格式  后来居然直接修改了my_lables.py文件 将里面的.jpg修改为.png就可以了

15.weights里面训练出的前1000每100存一次 后面都是10000才保存一次

修改darknet目录下examples文件夹里面的detector.c文件

将if（i%10000==0||(i<1000&&i%100==0))中的10000修改为1000就能每1000保存一次了

之后由于电脑问题 测试时间超过一分钟并且不出框就在别人电脑上运行的

# 心得

本次测试让我对Ubuntu系统有了更深刻的认识，并且相对来说更能沉住气、慢慢来。上网查找资料，自己常识，虽然说并不能一次成功，更甚至找不到有用的指令，还是要有更多的耐心阿。还有就是不要把任务全留到最后，一天做一点。









































