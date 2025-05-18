# MySQL

## 部署机器

- WSL2
- AutoDL的任何主机和**无卡开机模式**

## 安装

### 非Docker的命令行安装

#### 命令行

以下的安装方式适用于AutoDL和WSL2

```sh
# 更新apt-get工具
sudo apt-get update

# 安装mysql
sudo apt-get install mysql-server -y

# 开启mysql服务
sudo service mysql start

# 查看mysql是否运行
sudo service mysql status
```

**AutoDL上安装时出现的经过不用官，一切看sudo service mysql status**

AutoDL上的MySQL状态插叙如下：

```sh
root@autodl-container-7702429a5b-78acb290:~# sudo service mysql status
 * /usr/bin/mysqladmin  Ver 8.0.42-0ubuntu0.22.04.1 for Linux on x86_64 ((Ubuntu))
Copyright (c) 2000, 2025, Oracle and/or its affiliates.

Oracle is a registered trademark of Oracle Corporation and/or its
affiliates. Other names may be trademarks of their respective
owners.

Server version          8.0.42-0ubuntu0.22.04.1
Protocol version        10
Connection              Localhost via UNIX socket
UNIX socket             /var/run/mysqld/mysqld.sock
Uptime:                 57 sec

Threads: 2  Questions: 8  Slow queries: 0  Opens: 119  Flush tables: 3  Open tables: 38  Queries per second avg: 0.140
```

从这个结果可以得到的信息如下：

1. **服务已启动**
   - `Uptime: 57 sec` 表示 MySQL 已持续运行 57 秒。
   - `UNIX socket: /var/run/mysqld/mysqld.sock` 确认服务正在监听本地 socket。
2. **基本连接信息**
   - 版本：`8.0.42-0ubuntu0.22.04.1`（Ubuntu 官方维护的版本）。
   - 协议版本：`10`（MySQL 8.0 的标准协议）。
   - 连接方式：`Localhost via UNIX socket`（本地连接正常）。
3. **性能指标正常**
   - `Threads: 2`：活跃线程数正常。
   - `Open tables: 38`：当前打开的表数量合理。
   - 无慢查询（`Slow queries: 0`）。

WSL上的MySQL状态插叙如下：

```
(base) gx@DESKTOP-1OPLDI5:~$ sudo service mysql status
● mysql.service - MySQL Community Server
     Loaded: loaded (/lib/systemd/system/mysql.service; enabled; vendor preset: enabled)
     Active: active (running) since Tue 2025-05-06 16:37:57 CST; 3min 33s ago
   Main PID: 3730 (mysqld)
     Status: "Server is operational"
      Tasks: 37 (limit: 9390)
     Memory: 358.8M
     CGroup: /system.slice/mysql.service
             └─3730 /usr/sbin/mysqld
```

**1.服务状态解析**

- **Active: active (running)**
  ​	MySQL 服务当前处于 **运行状态**，且已持续 **3 分 33 秒**（自 `2025-05-06 16:37:57` 启动）。

- **Loaded: loaded (...; enabled)**
  服务已设置为 **开机自启**（`enabled`），系统重启后会自动启动 MySQL。

- **Status: "Server is operational"**
  MySQL 内部状态显示为正常运行，无严重错误。

**2. 关键进程信息**

- **主进程 PID**: `3730 (mysqld)`
  MySQL 主进程（`mysqld`）的进程 ID 为 3730，可通过以下命令查看详细信息：

- **资源占用**:
  - 内存：`358.8M`（初始运行时的正常占用，随查询量增长可能增加）。
  - 任务数：`37`（活跃线程数合理，无异常）。

#### 设置密码

在目前Linux版本的MySQL中，root用户使用系统用户身份进行登录和认证，没有数据库的密码，但是为了方便我们远程访问，这里进行密码设置。

```mysql
# 登录 Mysql 
sudo mysql
# root@autodl-container-7702429a5b-78acb290:~# sudo mysql
# Welcome to the MySQL monitor.  Commands end with ; or \g.

# 更改密码，替换新密码为123456
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '123456';
# mysql> ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '123456';
# Query OK, 0 rows affected (0.00 sec)

# 刷新权限
FLUSH PRIVILEGES;
# mysql> FLUSH PRIVILEGES;
# Query OK, 0 rows affected (0.00 sec)

# 退出
exit;
# mysql> exit;
# Bye
 
# 登录 Mysql “-u root” 是用root用户登录 “-p”是需要输入密码
sudo mysql -u root -p
# root@autodl-container-7702429a5b-78acb290:~# sudo mysql -u root -p
# Enter password: 
# Welcome to the MySQL monitor.  Commands end with ; or \g.
```

#### 修改配置

```sh
sudo vim /etc/mysql/mysql.conf.d/mysqld.cnf
```

将文件中的bind-address=127.0.0.1代码注释掉，或者改为0.0.0.0。

这里的含义是bind-address 参数限制了可以接受连接的 IP 地址，需要设置为允许从你的客户端 IP 访问。

改完需要重启MySQL服务

```sh
sudo systemctl restart mysql # wsl2使用此命令重启
service mysql restart # Autodl中使用了Docker，而非传统的systemd管理服务，所以使用词命令重启
```

### Docker安装

**老师推荐**

**拉取 MySQL 镜像**

```sh
docker pull mysql:latest # 最新版

# 或者指定版本（例如 MySQL 8.0） 或 5.7
docker pull mysql:8.0
```

**运行 MySQL 容器**

```sh
docker run -d \
  --name mysql-container \
  -e MYSQL_ROOT_PASSWORD=your_password \
  -p 3306:3306 \
  -v /path/on/host:/var/lib/mysql \
  mysql:latest
```

参数：

- `-d`：后台运行容器
- `--name`：容器名称（可自定义）
- `-e MYSQL_ROOT_PASSWORD`：设置 root 用户密码（必填）
- `-p 3306:3306`：将容器 3306 端口映射到宿主机 3306 端口
- `-v /path/on/host:/var/lib/mysql`：持久化数据存储（将宿主机目录挂载到容器）

**验证容器运行状态**

```sh
docker ps -a | grep mysql-container
```

  若看到状态为 `Up` 则表示运行成功。

**进入 MySQL 容器**

```text
docker exec -it mysql-container mysql -u root -p `your_password`
```

**常见问题解决**

- 端口冲突

若宿主机 3306 端口被占用，修改映射端口（例如 `-p 3307:3306`）。

- 数据持久化失败

确保挂载的宿主机目录有写入权限：

```text
chmod -R 777 /path/on/host
```

- 忘记 root 密码

重新启动容器并跳过权限验证：

```text
docker run -it --rm \
  -e MYSQL_ROOT_PASSWORD=temp_password \
  mysql:latest \
  mysqld --skip-grant-tables
```

然后通过容器内命令行重置密码。

**Docker控制MySQL的其他常用命令**

**停止容器**:

```text
docker stop mysql-container
```

**启动已停止的容器**:

```text
docker start mysql-container
```

**删除容器**:

```text
docker rm mysql-container
```

通过以上步骤，您已成功在 Docker 中部署 MySQL 并完成基础配置。根据实际需求调整参数即可！

## 远程访问

**注意如何你在WSL或WindowS本地也安装过MySQL，可能存在端口冲突的情况，可以在AutoDL上做端口映射的时候映射其它端口，或修改你WSL或WindowS的MySql端口**

### 访问WSL2上的MySQL

编写pymysql代码连接服务器

```python
# pip install pymysql

import pymysql

# 连接配置
connection = pymysql.connect(
    host='127.0.0.1',        # 本地地址
    port=3306,               # 本地映射的端口（与远程MySQL的3306绑定）
    user='root',             # MySQL 用户名
    password='123456',       # MySQL 密码
    database='',             # 可选数据库名
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

try:
    with connection.cursor() as cursor:
        # 执行一个简单的查询
        cursor.execute("SELECT VERSION()")
        result = cursor.fetchone()
        print("MySQL 版本:", result)
finally:
    connection.close()
```

结果

```sh
PS C:\Users\gx> & D:/python_Develop/anaconda3/envs/py310/python.exe "d:/gx/Desktop/cutting-edge technology/MySql/link.py"
MySQL 版本: {'VERSION()': '8.0.42-0ubuntu0.20.04.1'}
```

### 访问AutoDL上的MySQL

#### AutoDL上配置端口映射

参考：https://www.autodl.com/docs/ssh_proxy/

点击服务器的自定义服务

Windows用户请打开Powershell，Mac/Linux用户请打开终端，执行以下命令后回车：

```
ssh -CNg -L 6006:127.0.0.1:6006 root@connect.westc.gpuhub.com -p 47520
ssh -CNg -L [本地端口]:[目标主机]:[目标端口] 服务器的用户名@服务器IP -p 47520
```

如询问yes/no请回答yes，并输入以下密码(粘贴后密码不会显示是正常现象)，密码在这个连接下面

输入密码回车后无任何其他输出则为正常，如显示Permission denied则可能密码粘贴失 败，请手动输入密码(Win10终端易出现无法粘贴密码问题)

打开 [http://localhost:6006](http://localhost:6006/) 访问自定义服务

#### AutoDL上配置端口映射原理

总结一下上面这条命令做了什么：

**在本地启动一个 SSH 隧道，将本地的 `6006` 端口映射到远程服务器的 `6006` 端口，从而让你可以通过访问 `http://localhost:6006` 来使用远程服务器上运行的服务（如 TensorBoard）。 **

AutoDL 提供的端口映射功能，本质上是通过 **SSH 隧道（SSH Tunneling）** 实现本地与远程服务器之间的通信；这种技术可以将远程服务器上的某个服务“映射”到本地机器的一个端口上，使得你可以在本地浏览器或客户端访问该服务。

![](img\autodl_port_knowledge.png)

#### 访问MySQL

启动MySQL

```sh
# 开启MySQL服务
sudo service mysql start

#root@autodl-container-7702429a5b-78acb290:~# sudo service mysql start
# * Starting MySQL database server mysqld                                su: warning: #cannot change directory to /nonexistent: No such file or directory

# 查看mysql是否运行
sudo service mysql status
```

远程访问容器内的 MySQL（默认端口 `3306`），修改端口号即可：

```sh
ssh -CNg -L 6006:127.0.0.1:3306 root@connect.westc.gpuhub.com -p 47520
```

这样做的目的时将本地的 `6006` 映射到远程的 `127.0.0.1:3306`（即 MySQL 默认端口和地址）。

编写pymysql代码连接服务器

```sh
# pip install pymysql

import pymysql

# 连接配置
connection = pymysql.connect(
    host='127.0.0.1',        # 本地地址
    port=6006,               # 本地映射的端口（与远程MySQL的3306绑定）
    user='root',             # MySQL 用户名
    password='123456',       # MySQL 密码
    database='',             # 可选数据库名
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

try:
    with connection.cursor() as cursor:
        # 执行一个简单的查询
        cursor.execute("SELECT VERSION()")
        result = cursor.fetchone()
        print("MySQL 版本:", result)
finally:
    connection.close()
```

结果

```python
PS C:\Users\gx> & D:/python_Develop/anaconda3/envs/py310/python.exe "d:/gx/Desktop/cutting-edge technology/MySql/link.py"
MySQL 版本: {'VERSION()': '8.0.42-0ubuntu0.22.04.1'}
```



## 插入数据

### 数据介绍

插入下面表格的数据到MySQL数据库中

![](img\1746520816411.png)

### 实现步骤

1. 启动数据库服务
2. **创建数据库和表** ：首先需要在 MySQL 中创建一个数据库和一张表，用于存储这些数据。
3. **编写 Python 脚本** ：使用 `pymysql` 库连接到 MySQL，并将表格中的数据插入到表中。

### 创建数据库和表

连接数据库并启动，

- 执行建库命令：

```mysql
CREATE DATABASE IF NOT EXISTS student_scores;
# mysql> CREATE DATABASE IF NOT EXISTS student_scores;
# Query OK, 1 row affected (0.04 sec)

USE student_scores;
# mysql> USE student_scores;
# Database changed
```

- 执行建表命令

根据表格中的字段，创建一个名为 `student_scores` 的表。表结构如下：

| 字段名      | 类型    | 描述     |
| ----------- | ------- | -------- |
| id          | INT     | 学生编号 |
| name        | VARCHAR | 姓名     |
| chinese     | INT     | 语文成绩 |
| math        | INT     | 数学成绩 |
| english     | INT     | 英语成绩 |
| physics     | INT     | 物理成绩 |
| chemistry   | INT     | 化学成绩 |
| biology     | INT     | 生物成绩 |
| history     | INT     | 历史成绩 |
| geography   | INT     | 地理成绩 |
| politics    | INT     | 政治成绩 |
| total_score | INT     | 总分     |

创建表的 SQL 语句如下：

```mysql
CREATE TABLE IF NOT EXISTS student_scores (
    id INT PRIMARY KEY,
    name VARCHAR(255),
    chinese INT,
    math INT,
    english INT,
    physics INT,
    chemistry INT,
    biology INT,
    history INT,
    geography INT,
    politics INT,
    total_score INT
);

# Query OK, 0 rows affected (0.01 sec)
```

查看表结构

```mysql
DESCRIBE student_scores;

# mysql> DESCRIBE student_scores;
+-------------+--------------+------+-----+---------+-------+
| Field       | Type         | Null | Key | Default | Extra |
+-------------+--------------+------+-----+---------+-------+
| id          | int          | NO   | PRI | NULL    |       |
| name        | varchar(255) | YES  |     | NULL    |       |
| chinese     | int          | YES  |     | NULL    |       |
| math        | int          | YES  |     | NULL    |       |
| english     | int          | YES  |     | NULL    |       |
| physics     | int          | YES  |     | NULL    |       |
| chemistry   | int          | YES  |     | NULL    |       |
| biology     | int          | YES  |     | NULL    |       |
| history     | int          | YES  |     | NULL    |       |
| geography   | int          | YES  |     | NULL    |       |
| politics    | int          | YES  |     | NULL    |       |
| total_score | int          | YES  |     | NULL    |       |
+-------------+--------------+------+-----+---------+-------+
12 rows in set (0.01 sec)
```

### 编写python脚本

下面是一个完整的 Python 脚本，用于将表格中的数据插入到 MySQL 数据库中：

```python
# pip install pymysql

import pymysql

# 表格数据
data = [
    (1, '王小明', 85, 92, 88, 78, 82, 75, 80, 77, 83, 790),
    (2, '李华', 78, 89, 90, 85, 88, 82, 76, 84, 86, 818),
    (3, '张敏', 90, 83, 86, 91, 87, 88, 82, 85, 89, 841),
    (4, '陈刚', 82, 95, 80, 88, 90, 86, 83, 80, 81, 825),
    (5, '刘芳', 76, 82, 84, 79, 83, 78, 75, 77, 80, 774),
    (6, '杨威', 88, 90, 82, 92, 89, 87, 84, 86, 85, 843),
    (7, '吴静', 91, 86, 89, 84, 85, 85, 81, 88, 87, 824),
    (8, '赵鹏', 80, 93, 81, 87, 92, 84, 80, 83, 82, 812),
    (9, '孙悦', 79, 85, 87, 76, 80, 85, 77, 78, 81, 772),
    (10, '周琳', 84, 88, 91, 82, 86, 85, 83, 84, 88, 825),
    (11, '郑浩', 86, 91, 83, 89, 88, 86, 82, 85, 84, 836),
    (12, '冯雪', 77, 84, 88, 75, 81, 74, 76, 78, 80, 773),
    (13, '田甜', 92, 87, 85, 90, 89, 88, 84, 86, 87, 848),
    (14, '贺磊', 81, 94, 82, 86, 91, 85, 81, 82, 83, 815),
    (15, '钟莹', 78, 83, 86, 79,	 84, 77, 75, 78, 81, 771),
    (16, '姜涛', 87, 90, 84, 91, 88, 86, 83, 85, 84, 838),
    (17, '段丽', 90, 86, 89, 85, 87, 84, 82, 86, 88, 837),
    (18, '侯宇', 83, 92, 81, 88, 90, 87, 80, 83, 82, 816),
    (19, '袁梦', 76, 85, 88, 77, 82, 76, 78, 79, 80, 771),
    (20, '文轩', 88, 91, 85, 90, 89, 88, 84, 86, 87, 848)
] # 每一条记录是一个元组，对应表中的字段顺序。

# MySQL 连接配置
connection = pymysql.connect(
    host='127.0.0.1',
    port=6006,
    user='root',
    password='123456',
    database='student_scores',
    charset='utf8mb4'
)

try:
    with connection.cursor() as cursor:
        # 构造 SQL 语句
        sql = """
        INSERT INTO student_scores (
            id, name, chinese, math, english, physics, chemistry, biology, 
            history, geography, politics, total_score
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        # 批量插入数据
        cursor.executemany(sql, data) # 使用 executemany() 方法可以一次性插入多条记录，效率更高。
        
        # 提交事务 在执行插入操作后，调用 connection.commit() 提交事务，确保数据成功写入数据库。
        connection.commit() 
finally:
    connection.close()

print("数据插入成功！")
```

### 检查数据

进入数据库

```mysql
USE student_scores;
# mysql> USE student_scores;
# Database changed

SELECT * FROM student_scores;
# mysql> SELECT * FROM student_scores;
+----+-----------+---------+------+---------+---------+-----------+---------+---------+-----------+----------+-------------+
| id | name      | chinese | math | english | physics | chemistry | biology | history | geography | politics | total_score |
+----+-----------+---------+------+---------+---------+-----------+---------+---------+-----------+----------+-------------+
|  1 | 王小明    |      85 |   92 |      88 |      78 |        82 |      75 |      80 |        77 |       83 |         790 |
|  2 | 李华      |      78 |   89 |      90 |      85 |        88 |      82 |      76 |        84 |       86 |         818 |
|  3 | 张敏      |      90 |   83 |      86 |      91 |        87 |      88 |      82 |        85 |       89 |         841 |
|  4 | 陈刚      |      82 |   95 |      80 |      88 |        90 |      86 |      83 |        80 |       81 |         825 |
|  5 | 刘芳      |      76 |   82 |      84 |      79 |        83 |      78 |      75 |        77 |       80 |         774 |
|  6 | 杨威      |      88 |   90 |      82 |      92 |        89 |      87 |      84 |        86 |       85 |         843 |
|  7 | 吴静      |      91 |   86 |      89 |      84 |        85 |      85 |      81 |        88 |       87 |         824 |
|  8 | 赵鹏      |      80 |   93 |      81 |      87 |        92 |      84 |      80 |        83 |       82 |         812 |
|  9 | 孙悦      |      79 |   85 |      87 |      76 |        80 |      85 |      77 |        78 |       81 |         772 |
| 10 | 周琳      |      84 |   88 |      91 |      82 |        86 |      85 |      83 |        84 |       88 |         825 |
| 11 | 郑浩      |      86 |   91 |      83 |      89 |        88 |      86 |      82 |        85 |       84 |         836 |
| 12 | 冯雪      |      77 |   84 |      88 |      75 |        81 |      74 |      76 |        78 |       80 |         773 |
| 13 | 田甜      |      92 |   87 |      85 |      90 |        89 |      88 |      84 |        86 |       87 |         848 |
| 14 | 贺磊      |      81 |   94 |      82 |      86 |        91 |      85 |      81 |        82 |       83 |         815 |
| 15 | 钟莹      |      78 |   83 |      86 |      79 |        84 |      77 |      75 |        78 |       81 |         771 |
| 16 | 姜涛      |      87 |   90 |      84 |      91 |        88 |      86 |      83 |        85 |       84 |         838 |
| 17 | 段丽      |      90 |   86 |      89 |      85 |        87 |      84 |      82 |        86 |       88 |         837 |
| 18 | 侯宇      |      83 |   92 |      81 |      88 |        90 |      87 |      80 |        83 |       82 |         816 |
| 19 | 袁梦      |      76 |   85 |      88 |      77 |        82 |      76 |      78 |        79 |       80 |         771 |
| 20 | 文轩      |      88 |   91 |      85 |      90 |        89 |      88 |      84 |        86 |       87 |         848 |
+----+-----------+---------+------+---------+---------+-----------+---------+---------+-----------+----------+-------------+
20 rows in set (0.00 sec)
```

