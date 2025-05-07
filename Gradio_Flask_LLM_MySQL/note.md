# Gradio数据库查询

## 设备

- AutoDL CUDA12.1 **(因为要装VLLM)**
  - vllm推理Qwen3 8B  端口号 6006
  - mysql   端口号 3306
- windows
  - flask
  - Gradio

## 任务

- 使用 gradio + qwen3 + vllm 部署 text2sql 任务，插入3个表格的数据，并实现多表查询



## AutoDL端口映射

**端口映射前先启动mysql和vllm服务**

**数据库启动**

```sh
# 开启mysql服务
sudo service mysql start

# 查看mysql是否运行
sudo service mysql status
```

**vllm启动**

```sh
# pip install vllm

vllm serve  /root/autodl-tmp/Qwen/Qwen3-8B --served_model_name  Qwen3-8B --max-model-len 4096
```

参考：https://www.autodl.com/docs/ssh_proxy/

AutoDL可以直接使用一条命令完成多个端口在windows上的映射

```sh
ssh -CNg -L 8080:127.0.0.1:8000 -L 6006:127.0.0.1:3306 root@connect.westc.gpuhub.com -p 44064
```

- 第一个 -L 是映射vllm的服务到windows上
- 第二个 -L 是映射 mysql的服务到windows上

前面的端口是映射到windows上的端口号，而后面的端口是在AutoDL上启动服务后的端口号

## 数据操作

### 数据说明

现有三个数据分别为：**家庭联系表（data/home.txt）**、**身体状况表：（data/body.txt）**、**学生成绩表（data/scores.txt）**

### 数据库操作

**数据插入**

创建数据库,名字为：student_info

```sh
# 在 mysql> 操作

CREATE DATABASE student_info;
use student_info
```

创建数据表，名字为：

```sh
# 在 mysql> 操作

CREATE TABLE family (
    id INT PRIMARY KEY,
    father_name VARCHAR(50),
    mother_name VARCHAR(50),
    father_phone VARCHAR(20),
    mother_phone VARCHAR(20)
);

CREATE TABLE body (
    id INT PRIMARY KEY,
    height FLOAT,
    weight FLOAT,
    age INT,
    gender VARCHAR(10),
    body_fat FLOAT,
    blood_sugar FLOAT
);


CREATE TABLE scores (
    id INT PRIMARY KEY,
    name VARCHAR(50),
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
```

建表后查询一下：

```sh
mysql> SHOW TABLES;
+------------------------+
| Tables_in_student_info |
+------------------------+
| body                   |
| family                 |
| scores                 |
+------------------------+
3 rows in set (0.00 sec)
```

deepseek撰写的mysql插入代码

```python
import pymysql
import pandas as pd

# 数据库连接配置
db_config = {
    'host': 'localhost',
    'port': 6006,  # AutoDL的服务器
     # 'port': 3306,  # AutoDL的服务器
    'user': 'root',
    'password': '123456',
    'charset': 'utf8mb4'
}

def import_data():
    """导入数据到数据库"""
    try:
        conn = pymysql.connect(**db_config, database='student_info')
        
        # 导入家庭联系数据
        family_data = pd.read_csv('data/home.txt', sep='\t')
        with conn.cursor() as cursor:
            # 先清空表（可选）
            cursor.execute("TRUNCATE TABLE family")
            for _, row in family_data.iterrows():
                cursor.execute(
                    "INSERT INTO family VALUES (%s, %s, %s, %s, %s)",
                    (row['序号'], row['父亲姓名'], row['母亲姓名'], row['父亲电话'], row['母亲电话'])
                )
        
        # 导入身体状况数据
        body_data = pd.read_csv('data/body.txt', sep='\t')
        with conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE body")
            for _, row in body_data.iterrows():
                cursor.execute(
                    "INSERT INTO body VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (row['序号'], row['身高（cm）'], row['体重（kg）'], row['年龄（岁）'], 
                     row['性别'], row['体脂（%）'], row['血糖（mmol/L）'])
                )
        
        # 导入学生成绩数据
        scores_data = pd.read_csv('data/scores.txt', sep='\t')
        with conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE scores")
            for _, row in scores_data.iterrows():
                cursor.execute(
                    "INSERT INTO scores VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (row['序号'], row['姓名'], row['语文'], row['数学'], row['英语'], 
                     row['物理'], row['化学'], row['生物'], row['历史'], row['地理'], 
                     row['政治'], row['总分'])
                )
        
        conn.commit()
        print("数据导入成功")
    except Exception as e:
        conn.rollback()
        print(f"数据导入失败: {e}")

# 执行函数
if __name__ == "__main__":
    import_data()
```

插入完成后查看数据库中的数据：

```sh
mysql> SELECT * FROM family;
+----+-------------+-------------+--------------+--------------+
| id | father_name | mother_name | father_phone | mother_phone |
+----+-------------+-------------+--------------+--------------+
|  1 | 王强        | 李梅        | 13800138000  | 13900139000  |
|  2 | 李军        | 张华        | 13600136000  | 13700137000  |
|  3 | 张勇        | 陈丽        | 13500135000  | 13400134000  |
|  4 | 陈一刚      | 周敏        | 13200132000  | 13300133000  |
|  5 | 刘辉        | 杨娟        | 13100131000  | 13000130000  |
|  6 | 杨明        | 赵芳        | 14700147000  | 14800148000  |
|  7 | 吴俊        | 孙燕        | 14900149000  | 15000150000  |
|  8 | 赵刚        | 钱丽        | 15100151000  | 15200152000  |
|  9 | 孙林        | 何娜        | 15300153000  | 15400154000  |
| 10 | 周建        | 徐慧        | 15500155000  | 15600156000  |
| 11 | 郑华        | 马丽        | 15700157000  | 15800158000  |
| 12 | 冯峰        | 朱婷        | 15900159000  | 16000160000  |
| 13 | 田军        | 高敏        | 16100161000  | 16200162000  |
| 14 | 贺伟        | 郭玲        | 16300163000  | 16400164000  |
| 15 | 钟明        | 罗霞        | 16500165000  | 16600166000  |
| 16 | 姜云涛      | 唐瑶        | 16700167000  | 16800168000  |
| 17 | 段勇        | 谢红        | 16900169000  | 17000170000  |
| 18 | 侯军        | 卢芳        | 17100171000  | 17200172000  |
| 19 | 袁刚        | 黄琴        | 17300173000  | 17400174000  |
| 20 | 文辉        | 吴兰        | 17500175000  | 17600176000  |
+----+-------------+-------------+--------------+--------------+
20 rows in set (0.00 sec)

mysql> SELECT * FROM body;
+----+--------+--------+------+--------+----------+-------------+
| id | height | weight | age  | gender | body_fat | blood_sugar |
+----+--------+--------+------+--------+----------+-------------+
|  1 |    175 |     70 |   20 | 男     |       18 |         5.5 |
|  2 |    168 |     55 |   19 | 女     |       22 |         5.2 |
|  3 |    180 |     75 |   21 | 男     |       16 |         5.8 |
|  4 |    172 |     68 |   20 | 男     |       17 |         5.3 |
|  5 |    165 |     52 |   18 | 女     |       20 |           5 |
|  6 |    178 |     73 |   22 | 男     |       19 |         5.6 |
|  7 |    160 |     50 |   19 | 女     |       21 |         5.1 |
|  8 |    173 |     66 |   20 | 男     |       18 |         5.4 |
|  9 |    170 |     64 |   21 | 男     |       19 |         5.7 |
| 10 |    162 |     53 |   18 | 女     |       20 |         5.2 |
| 11 |    176 |     71 |   22 | 男     |       18 |         5.5 |
| 12 |    166 |     56 |   19 | 女     |       21 |         5.3 |
| 13 |    182 |     78 |   23 | 男     |       17 |         5.9 |
| 14 |    174 |     69 |   20 | 男     |       18 |         5.4 |
| 15 |    164 |     51 |   18 | 女     |       20 |           5 |
| 16 |    177 |     72 |   21 | 男     |       19 |         5.6 |
| 17 |    163 |     54 |   19 | 女     |       21 |         5.2 |
| 18 |    171 |     67 |   20 | 男     |       18 |         5.3 |
| 19 |    167 |     58 |   18 | 女     |       20 |         5.1 |
| 20 |    179 |     74 |   22 | 男     |       17 |         5.7 |
+----+--------+--------+------+--------+----------+-------------+
20 rows in set (0.00 sec)

mysql> SELECT * FROM scores;
+----+-----------+---------+------+---------+---------+-----------+---------+---------+-----------+----------+-------------+
| id | name      | chinese | math | english | physics | chemistry | biology | history | geography | politics | total_score |
+----+-----------+---------+------+---------+---------+-----------+---------+---------+-----------+----------+-------------+
|  1 | 王小明    |      85 |   92 |      88 |      78 |        82 |      75 |      80 |        77 |       83 |         790 |
|  2 | 李华      |      78 |   89 |      90 |      85 |        88 |      82 |      76 |        84 |       86 |         818 |
|  3 | 张敏      |      90 |   83 |      86 |      91 |        87 |      88 |      82 |        85 |       89 |         841 |
|  4 | 陈刚      |      82 |   95 |      80 |      88 |        90 |      86 |      83 |        80 |       81 |         825 |
|  5 | 刘芳      |      76 |   82 |      84 |      79 |        83 |      78 |      75 |        77 |       80 |         774 |
|  6 | 杨威      |      88 |   90 |      82 |      92 |        89 |      87 |      84 |        86 |       85 |         843 |
|  7 | 吴静      |      91 |   86 |      89 |      84 |        85 |      83 |      81 |        88 |       87 |         824 |
|  8 | 赵鹏      |      80 |   93 |      81 |      87 |        92 |      84 |      80 |        83 |       82 |         812 |
|  9 | 孙悦      |      79 |   85 |      87 |      76 |        80 |      77 |      78 |        79 |       81 |         772 |
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

## 模型部署

模型下载

Qwen3 8B  效果非常的好

```python
# 模型下载   
# pip install modelscope
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-8B', cache_dir="/root/autodl-tmp/")
```

**Qwen3 模型会在回复前进行思考。这种行为可以通过硬开关（完全禁用思考）或软开关（模型遵循用户关于是否应该思考的指令）来控制。**

**硬开关在 vLLM 中可以通过以下 API 调用配置使用。要禁用思考，请在client.chat.completions.create中设置： extra_body={"chat_template_kwargs": {"enable_thinking": False}}**

调用代码：

~~~python
def text_2_sql(text):

    prompt = f"""假设你是一个资深的MySQL数据库专家，你能将我给你提交的任务转换成MySQL语句并输出。
请注意：
1. 所有的操作均在名字为student_scores的数据库和student_scores的表上进行。
2. MySQL数据库共有 id、name、chinese、math、english、physics、chemistry、biology、history、geography、politics、total_score共计12列。
3. 拒绝所有的增加、删除和修改操作，如果用户提出了这三个操作，直接返回拒绝；
样例如下：
```
用户输入：请帮查找出在名字为王小明的语文成绩是多少？
输出：SELECT chinese FROM student_scores WHERE name = '王小明';
```
```
用户输入：请帮查找出英语成绩大于60分的学生名字都有谁？
输出：SELECT name FROM student_scores WHERE english > 60;
```
```
用户输入：请帮查找出数学成绩小于90分的学生学号是什么？
输出：SELECT id FROM student_scores WHERE math < 90;
```
```
用户输入：请查找出英语成绩大于70分的学生的数学成绩是多少？
输出：SELECT math FROM student_scores WHERE english > 70;
```
```
用户输入：{text}。
输出：
```
"""

    chat_response = client.chat.completions.create(
        model= model_name, # "Qwen3-8B",
        temperature=0.5,
        top_p=0.7,
        presence_penalty=1.5,
    	extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        messages=[
            {"role": "system", "content": "你是一个数据库助手"},
            {"role": "user", "content": prompt},
        ],
    )
    sql = chat_response.choices[0].message.content

    # 使用正则表达式提取 SQL 语句
    match = re.search(r'```(?:sql)?\n(.*?)\n```', sql, re.DOTALL)

    if match:
        sql_query = match.group(1).strip()

        return sql_query
    
    else:
        return sql
~~~

## flask

运行的地址为： http://127.0.0.1:5000

flask作为服务端，其代码如下所示：

~~~python
from openai import OpenAI
import re
import pymysql
from flask import Flask, jsonify, request

# 创建 Flask 应用
app = Flask(__name__)

def text_2_sql(text):

    prompt = f"""假设你是一个资深的MySQL数据库专家，你能将我给你提交的任务转换成MySQL语句并输出。
请注意：
1. 所有的操作只在名字为student_info的数据库和scores、family、body的表上进行。
2. scores表共有 id、name、chinese、math、english、physics、chemistry、biology、history、geography、politics、total_score共计12列。
3. family表共有 id、father_name、mother_name、father_phone、mother_phone 共计5列。
4. body表共有 id、height、weight、age、gender、body_fat、blood_sugar 共计7列。
5. 三张表均有id字段，id字段是primary key，可用于多表关联查询。
6. 拒绝所有的增加、删除和修改操作，如果用户提出了这三个操作，直接返回拒绝；
样例如下：
```
用户输入：请帮查找出在名字为王小明的语文成绩是多少？
输出：SELECT chinese FROM scores WHERE name = '王小明';
```
```
用户输入：查找出身高大于170cm的学生姓名和父亲电话。
输出：SELECT s.name, f.father_phone FROM scores s JOIN body p ON s.id = p.id JOIN family f ON s.id = f.id WHERE p.height > 170;
```
```
用户输入：查找出体重大于60kg的学生姓名和母亲姓名。
输出：SELECT s.name, f.mother_name FROM scores s JOIN body p ON s.id = p.id JOIN family f ON s.id = f.id WHERE p.weight > 60;
```
```
用户输入：请帮查找出英语成绩大于60分的学生名字都有谁？
输出：SELECT name FROM scores WHERE english > 60;
```
```
用户输入：请帮查找出数学成绩小于90分的学生学号是什么？
输出：SELECT id FROM scores WHERE math < 90;
```
```
用户输入：请查找出英语成绩大于70分的学生的数学成绩是多少？
输出：SELECT math FROM scores WHERE english > 70;
```
```
用户输入：{text}。
输出：
```
"""


    chat_response = client.chat.completions.create(
        model= model_name, # "Qwen3-8B",
        temperature=0.5,
        top_p=0.7,
        presence_penalty=1.5,
    	extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        messages=[
            {"role": "system", "content": "你是一个数据库助手"},
            {"role": "user", "content": prompt},
        ],
    )
    sql = chat_response.choices[0].message.content
    print(sql)

    # 使用正则表达式提取 SQL 语句
    match = re.search(r'```(?:sql)?\n(.*?)\n```', sql, re.DOTALL)

    if match:
        sql_query = match.group(1).strip()

        return sql_query
    
    else:
        return sql

def get_sql_result(sql):
    # 数据库的操作简易加上异常吃力
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            return cursor.fetchall()
    except Exception as e:
        return f"查询错误: {str(e)}"

    return "查询错误"

@app.route("/student", methods=["POST"])
def get_result():

    data = request.get_json()
    print(data)

    natural_language = data.get("text", "").strip()
    print(natural_language)

    if not natural_language:
        return jsonify({"error": "缺少查询内容"}), 400

    # qwen text -> sql
    sql = text_2_sql(natural_language)
    print(sql)
    
    # 执行 sql
    result = get_sql_result(sql) # 结果可以是一个字符 也可以是一个表格
    print("查询结果", result)


    # 使用 jsonify 包装结果，确保返回合法的 HTTP 响应
    return jsonify({
        "sql": str(sql),
        "result": str(result)
    })


if __name__ == "__main__":

    # vllm设置
    model_name = "Qwen3-8B"
    openai_api_key = "EMPTY"
    openai_api_base = "http://127.0.0.1:8080/v1"
    client = OpenAI(
        base_url=openai_api_base,
        api_key=openai_api_key
    )

    # mysql数据库连接配置
    connection = pymysql.connect(
        host='127.0.0.1',        # 本地地址
        # port=3306,               # WSL端口
        port=6006,               # Autodl端口
        user='root',             # MySQL 用户名
        password='123456',       # MySQL 密码
        database='student_info',      # 要操作的数据库名
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    
    # 应用启动
    app.run(debug=True)
    # 请帮查找出在名字为王小明的语文成绩是多少？
~~~

## Gradio

Gradio作为客户端，其代码如下所示：

```python
import gradio as gr
import requests

def main(query):
    query_data = {
        "text": query,
        "other": "其它信息" 
    }
    
    response = requests.post(  
        url="http://127.0.0.1:5000/student",
        json=query_data,
        headers={"Content-Type": "application/json"}  
    )

    if response.status_code == 200:
        data = response.json()
        result = "".join(data["result"])
        sql = data["sql"]
    else:
        result = "查询失败"
        sql = "无"

    # 流式输出 result
    accumulated_result = "查询的结果为：\n"
    for char in result:
        accumulated_result += char
        yield "", accumulated_result
        # 模拟流式效果
        time.sleep(0.01)

    # 查询完成后一次性输出 SQL
    yield sql, accumulated_result


# 启动服务
if __name__ == "__main__":
    import time

    with gr.Blocks() as demo:
        gr.Markdown("## 🧾 数据库信息查询系统\n请输入你的查询需求")

        with gr.Row():
            input_box = gr.Textbox(label="输入查询要求", placeholder="请描述你要查询的内容...")

        # 添加一个按钮
        btn = gr.Button("🔍 生成结果")

        with gr.Tab("查询结果"):
            result_output = gr.Textbox(label="查询结果（逐字输出）", lines=10)
        
        with gr.Tab("SQL 语句"):
            sql_output = gr.Textbox(label="实际执行的 SQL 语句", lines=10)

        examples = gr.Examples(
            examples=[
                ["查找出体重大于60kg的学生姓名和母亲姓名。"], 
                ["查找出身高大于170cm的学生姓名和父亲电话。"],
                ["所有学生的数学平均分是多少？"]
            ],
            inputs=input_box
        )

        # 绑定按钮点击事件
        btn.click(fn=main, inputs=input_box, outputs=[sql_output, result_output])

    demo.launch(
        server_name="127.0.0.1",
        server_port=55555
    )
```

## 运行机制

**Gradio**

我们使用Gradio创建页面并作为项目的客户端与客户进行功能交互：

Gradio收集用户的问题等，封装成json的形式传递给Flask服务端去处理（http传输文本需要使用json的格式进行传输）；受到Flask服务端处理后的结果在Gradio客户端上进行显示；

**Flask**

flask的作用是把模型和数据库的服务变成网络接口让别人访问

我们使用flask作为项目的服务端，在Flask上使用部署的大模型根据要求生成sql，并执行sql查询得到记过，并将结果返回给Gradio客户端；

## 运行测试

查询的问题：查找出体重大于60kg的学生姓名和母亲姓名，其运行效果如下：

![](img\1.png)

flask服务端的部分运行日志：

```sh
 * Detected change in 'D:\\gx\\Desktop\\cutting-edge technology\\Gradio_Flask_LLM_MySQL\\src\\service.py', reloading
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 584-536-653
PS D:\gx\Desktop\cutting-edge technology\Gradio_Flask_LLM_MySQL\src> python .\service.py
 * Serving Flask app 'service'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 584-536-653
{'text': '查找出体重大于60kg的学生姓名和母亲姓名。', 'other': '其它信息'}
查找出体重大于60kg的学生姓名和母亲姓名。
SELECT s.name, f.mother_name FROM scores s JOIN body p ON s.id = p.id JOIN family f ON s.id = f.id WHERE p.weight 
> 60;
SELECT s.name, f.mother_name FROM scores s JOIN body p ON s.id = p.id JOIN family f ON s.id = f.id WHERE p.weight 
> 60;
查询结果 [{'name': '王小明', 'mother_name': '李梅'}, {'name': '张敏', 'mother_name': '陈丽'}, {'name': '陈刚', 'mother_name': '周敏'}, {'name': '杨威', 'mother_name': '赵芳'}, {'name': '赵鹏', 'mother_name': '钱丽'}, {'name': ' 
孙悦', 'mother_name': '何娜'}, {'name': '郑浩', 'mother_name': '马丽'}, {'name': '田甜', 'mother_name': '高敏'}, {'name': '贺磊', 'mother_name': '郭玲'}, {'name': '姜涛', 'mother_name': '唐瑶'}, {'name': '侯宇', 'mother_name': '卢芳'}, {'name': '文轩', 'mother_name': '吴兰'}]
127.0.0.1 - - [07/May/2025 21:30:41] "POST /student HTTP/1.1" 200 -
```

Graido的部分运行日志：

```sh
PS D:\gx\Desktop\cutting-edge technology\Gradio_Flask_LLM_MySQL\src> python .\client.py
* Running on local URL:  http://127.0.0.1:55555

* Running on local URL:  http://127.0.0.1:55555

To create a public link, set `share=True` in `launch()`.
<Response [200]>
[{'name': '王小明', 'mother_name': '李梅'}, {'name': '张敏', 'mother_name': '陈丽'}, {'name': '陈刚', 'mother_name': '周敏'}, {'name': '杨威', 'mother_name': '赵芳'}, {'name': '赵鹏', 'mother_name': '钱丽'}, {'name': '孙悦', 'mother_name': '何娜'}, {'name': '郑浩', 'mother_name': '马丽'}, {'name': '田甜', 'mother_name': '高敏'}, {'name': ' 贺磊', 'mother_name': '郭玲'}, {'name': '姜涛', 'mother_name': '唐瑶'}, {'name': '侯宇', 'mother_name': '卢芳'}, {'name': '文轩', 'mother_name': '吴兰'}]
```





