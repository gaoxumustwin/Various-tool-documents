# wsl配置

## 参考

https://blog.csdn.net/qq_40102732/article/details/135182310?ops_request_misc=&request_id=&biz_id=102&utm_term=wsl%20%E5%AE%89%E8%A3%85cuda&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-3-135182310.142^v100^pc_search_result_base8&spm=1018.2226.3001.4187#%E4%BA%8C-%E5%AE%89%E8%A3%85cuda-cudnn-TensorRT_3

## WSL安装

https://blog.csdn.net/x777777x/article/details/141092913?spm=1001.2014.3001.5506

## WSL安装miniconda

https://blog.csdn.net/youhebuke225/article/details/135211461?spm=1001.2014.3001.5506

## WSL更换国内源

第一步：备份源文件：

```sh
sudo cp /etc/apt/sources.list /etc/apt/sources.list.backup
```

第二步：编辑/etc/apt/sources.list文件
在文件最前面添加以下条目(操作前请做好相应备份)：

```sh
vi /etc/apt/sources.list
```

**阿里云源**

```sh
deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-proposed main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse
```

**清华源**

```s
#默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
#deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
#deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
#deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse
#deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-security main restricted universe multiverse
#预发布软件源，不建议启用
#deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
#deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-proposed main restricted universe multiverse
```

第三步：执行更新命令：

```sh
sudo apt-get update
sudo apt-get upgrade
```

## WSL安装Docker 

在Linux系统下，安装 Docker 也算是比较简单的操作，我们可以运行 Docker 的便捷安装脚本：

```
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
```

在输入这两行命令后，Docker 即安装完毕，我们只需要启动它并赋予他权限即可：

```
sudo systemctl enable docker
sudo systemctl start docker
sudo chmod 777 /var/run/docker.sock
sudo systemctl restart docker
```

如果一切都安装完毕，你可以在终端中输入下列命令进行验证。

```
docker run hello-world
```

若一切正常，你将会看到类似如下显示：

```
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
719385e32844: Pull complete 
Digest: sha256:fc6cf906cbfa013e80938cdf0bb199fbdbb86d6e3e013783e5a766f50f5dbce0
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.
```

如果失败出现下面：

```
Unable to find image 'hello-world:latest' locally
docker: Error response from daemon: Get "https://registry-1.docker.io/v2/": net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers).
See 'docker run --help'.
```

请参考：

https://blog.csdn.net/Liiiiiiiiiii19/article/details/142438122?ops_request_misc=&request_id=&biz_id=102&utm_term=docker:%20Error%20response%20from%20da&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-142438122.142^v100^pc_search_result_base4&spm=1018.2226.3001.4187

```
{
"registry-mirrors": ["https://docker.registry.cyou",
"https://docker-cf.registry.cyou",
"https://dockercf.jsdelivr.fyi",
"https://docker.jsdelivr.fyi",
"https://dockertest.jsdelivr.fyi",
"https://mirror.aliyuncs.com",
"https://dockerproxy.com",
"https://mirror.baidubce.com",
"https://docker.m.daocloud.io",
"https://docker.nju.edu.cn",
"https://docker.mirrors.sjtug.sjtu.edu.cn",
"https://docker.mirrors.ustc.edu.cn",
"https://mirror.iscas.ac.cn",
"https://docker.rainbond.cc"]
}
```

## WSL配置C++环境

### 安装gcc和g++

```sh
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-9 g++-9 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
```

其中最后一句会把当前的默认GCC版本设置为gcc-9。

尽量不要安装gcc 11以上的版本，在cuda编程的时候出现以下的问题：

```sh
In file included from /usr/local/cuda-11.7/bin/../targets/x86_64-linux/include/cuda_runtime.h:83,
                 from <command-line>:
/usr/local/cuda-11.7/bin/../targets/x86_64-linux/include/crt/host_config.h:132:2: error: #error -- unsupported GNU version! gcc versions later than 11 are not supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.
  132 | #error -- unsupported GNU version! gcc versions later than 11 are not supported! The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.
```

**安装GCC-10.0**

```
sudo add-apt-repository ppa:ubuntu-toolchain-r/ppa
sudo apt update
sudo apt install gcc-10 g++-10 
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 60 --slave /usr/bin/g++ g++ /usr/bin/g++-10
```

### 安装CMake

命令行安装cmake

```sh
sudo apt-get install cmake
```

编译安装cmake

```sh
git clone https://github.com/Kitware/CMake.git
cd CMake
./bootstrap
make -j8
sudo make install
cmake --version
```

### 安装protobuf

```sh
sudo apt-get install autoconf automake libtool curl make g++ unzip
git clone https://github.com/google/protobuf.git
cd protobuf
git submodule update --init --recursive  # 可能会失败，最好网络代理
./autogen.sh
./configure
make -j8
make check
sudo make install
sudo ldconfig # refresh shared library cache.
```

然后查看版本：

```sh
$ protoc --version
libprotoc 3.17.2
```



## WSL安装CUDA

下载地址：https://developer.nvidia.com/cuda-11-7-0-download-archive

- 安装

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-wsl-ubuntu-11-7-local_11.7.0-1_amd64.deb # 如果下载慢 可以将连接单独复制到浏览器上去下载
sudo dpkg -i cuda-repo-wsl-ubuntu-11-7-local_11.7.0-1_amd64.deb
# 注意下面这句话 * 这个地方根据自己的情况来 当执行上面的命令时 会出现这个*的具体值
sudo cp /var/cuda-repo-wsl-ubuntu-11-7-local/cuda- * -keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

注意：（参考：https://zhuanlan.zhihu.com/p/703912534）

安装cuda时，出现以下错误：

```
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
Some packages could not be installed. This may mean that you have
requested an impossible situation or if you are using the unstable
distribution that some required packages have not yet been created
or been moved out of Incoming.
The following information may help to resolve the situation:

The following packages have unmet dependencies:
 nsight-systems-2023.3.3 : Depends: libtinfo5 but it is not installable
E: Unable to correct problems, you have held broken pack
```

解决方案：

1.打开镜像源文件，并修改。

```text
sudo vim /etc/apt/sources.list.d/ubuntu.sources
```

2.向sources文件中添加如下内容：

```text
Types: deb
URIs: http://archive.ubuntu.com/ubuntu/
Suites: lunar
Components: universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
```

3.更新

```text
sudo apt-get update
```

4.重新安装cuda

```text
 sudo apt install cuda -y
```

- 环境配置

配置CUDA的环境变量：

```sh
sudo vim ~/.bashrc 
```

.bashrc文件末尾添加：

```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.7/lib64
export PATH=$PATH:/usr/local/cuda-11.7/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-11.7
export PATH=$PATH:/usr/local/cuda/bin


# 直接在终端运行
sudo touch /etc/profile.d/cuda.sh
echo 'export PATH=/usr/local/cuda/bin/:$PATH' | sudo tee -a /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/lib/wsl/lib/:$LD_LIBRARY_PATH' | sudo tee -a /etc/profile.d/cuda.sh
 
# 这个会根据vim ~/.bsahrc去编辑，再source ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-11.7
```

更新文件：

```sh
source ~/.bashrc
```

- 验证安装

```sh
nvcc --version
```

- 代码测试

编写一个名字为test.cu的代码

```c++
#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
using namespace std;
int main() {
    int count = 0;
 	cudaGetDeviceCount(&count);
	cout <<"当前计算机包含GPU数为"<< count << endl;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) 
	    printf("%s\n", cudaGetErrorString(err));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device Number: %d\n", 0);
    cout << "当前设备名字为" << prop.name << endl;
	cout << "GPU全局内存总量为" << prop.totalGlobalMem << endl;
	cout << "单个线程块中包含的线程数最多为" << prop.maxThreadsPerBlock << endl;
}

// from https://blog.csdn.net/chongbin007/article/details/123973475
/*
打印结果如下：
当前计算机包含GPU数为1
Device Number: 0
当前设备名字为NVIDIA GeForce RTX 3060
GPU全局内存总量为12884377600
单个线程块中包含的线程数最多为1024
*/
```

在命令行中执行：

```sh
nvcc test.cu -o test
./test
```

- cuda卸载

```sh
sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*" "*npp*" "*nvjpeg*" "nsight*" -y

sudo apt-get autoremove -y
```



## WSL安装cuDNN

进入https://developer.nvidia.com/rdp/cudnn-archive去选择版本，并下载Local Installer for Linux86_64(Tar)  需要登录账户

将下载后的tar包复制到文件夹下，通过解压安装，然后进行配置即可。例如我下载下来的是：cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz

```sh
tar -xvf cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
sudo cp cudnn-linux-x86_64-8.5.0.96_cuda11-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cudnn-linux-x86_64-8.5.0.96_cuda11-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

验证是否安装成功，输入如下命令，显示如下图即安装成功

```sh
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

显示的数字与tar包显示一致，安装成功



## WSL安装TensorRT

TensorRT选择对应的tar包进行安装

- 下载

地址： https://developer.nvidia.com/nvidia-tensorrt-8x-download

选择8.6 GA，注意到越网上对应的cuda版本越高

- 安装

将下载后的tar包复制到文件夹下，通过解压安装，然后进行配置即可。安装完成后会出现TensorRT-(版本)的文件夹，这里我们已经安装好了

```sh
tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
```

将TensorRT 下的lib绝对路径添加到系统环境中(根据自己的安装目录来)

```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/e/ubuntu_20.04/TensorRT-8.6.1.6/lib
```

创建TensorRT的python虚拟环境

切换到python目录下，安装tensorrt python whl文件
注意：这里的pip安装一定是下载到自己的虚拟环境下，比如anaconda下pytorch环境激活后
根据当前环境的python版本安装对应的tensorrt, 我的当前环境python为3.9，就安装3.9对应的tensorrt whl文件

```sh
conda create -n TensorRT python=3.9 -y
conda activate TensorRT

# worspace  TensorRT-8.6.1.6
# 安装tensorrt python whl文件
cd python
pip install tensorrt-8.6.1-cp39-none-linux_x86_64.whl

# 安装uff whl
cd ../uff
pip install uff-0.6.9-py2.py3-none-any.whl
 
# 安装graphsurgeon whl
cd ../graphsurgeon
pip install graphsurgeon-0.4.6-py2.py3-none-any.whl
 
# 安装onnx_graphsurgeon whl
cd ../onnx_graphsurgeon
pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
```

- python tesnorrt 验证

```sh
$python
>>import tensorrt
>>print(tensorrt.__version__)
>>assert tensorrt.Builder(tensorrt.Logger())
```

- 验证安装

查看tensorRT安装版本

```sh
sudo find / -name NvInferVersion.h
```

- 运行测试案例

切换到samples/sampleOnnxMNIST文件夹下编译，然后会在 **TensoRT-{版本}/targets/x86_64-linux-gnu/bin/** 文件夹下得到一个sample_onnx_mnist的可执行文件，运行

```sh
# worspace  TensorRT-8.6.1.6
# 切换到TensorRT-{版本}/samples/sampleOnnxMNIST/目录下编译
cd samples/sampleOnnxMNIST/
make
# 切换到TensorRT-{版本}/targets/x86_64-linux-gnu/bin目录下运行sample_onnx_mnist
cd ../../targets/x86_64-linux-gnu/bin/
./sample_onnx_mnist
```



## WSL安装torch、torchvision、torchaudio安装

最后我们在虚拟环境内安装的torch、torchvision、以及torchaudio也要保证cuda与上面3个保持一致性,去pytorch whl网站去下载对应版本的文件https://download.pytorch.org/whl/torch_stable.html，然后进行安装

```sh
# torch安装
pip install torch-1.12.0+cu116-cp39-cp39-linux_x86_64.whl 
 
# torchaudio安装
pip install torchaudio-0.12.0+cu116-cp39-cp39-linux_x86_64.whl
 
# torchvision安装
pip install torchvision-0.13.0+cu116-cp39-cp39-linux_x86_64.whl
```



## WSL安装milvus

具体内容参考对应文件夹



## WSL安装neo4j

具体内容参考对应文件夹