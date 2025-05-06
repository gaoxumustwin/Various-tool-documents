# 大模型数据库查询

## 设备

- AutoDL CUDA12.1
  - vllm推理Qwen3 8B  端口号：
  - mysql   端口号 3306
- windows docker
  - dify

## 任务

- 安装mysql + pymysql 插入数据
- 部署qwen3-0.6B
- 接收用户输入的中文需求并使用 qwen3 转化为 sql 语句
- 使用 fask 部署 ap 接口，该 api 用于 执行 sql 语句，并将结果返回
- 使用 dify 工作流模式，调用 fask 部署的 api，将以上工作流进行串联

## 数据库操作

参考MySQL文件

数据存储在数据库名字为：student_scores

数据表名字为：student_scores

**数据库启动**

```sh
# 开启mysql服务
sudo service mysql start

# 查看mysql是否运行
sudo service mysql status
```

## 模型部署

模型

Qwen3 8B

```python
# 模型下载   
# pip install modelscope
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-8B', cache_dir="/root/autodl-tmp/")
```

vllm启动

```sh
# pip install vllm

vllm serve /mnt/e/model/Qwen3-0___6B --served_model_name  Qwen3-0.6B --max-model-len 1024
# --host 0.0.0.0  --port 8080 

vllm serve  /root/autodl-tmp/Qwen/Qwen3-8B --served_model_name  Qwen3-8B --max-model-len 1024 
```

**Qwen3 模型会在回复前进行思考。这种行为可以通过硬开关（完全禁用思考）或软开关（模型遵循用户关于是否应该思考的指令）来控制。**

**硬开关在 vLLM 中可以通过以下 API 调用配置使用。要禁用思考，请在client.chat.completions.create中设置： extra_body={"chat_template_kwargs": {"enable_thinking": False}}**

例如：

```python
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {"role": "user", "content": "Give me a short introduction to large language models."},
    ],
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    max_tokens=8192,
    presence_penalty=1.5,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print("Chat response:", chat_response)
```

## AutoDL端口映射

参考：https://www.autodl.com/docs/ssh_proxy/

AutoDL可以直接使用一条命令完成多个端口在windows上的映射

```sh
ssh -CNg -L 8080:127.0.0.1:8000 -L 6006:127.0.0.1:3306 root@connect.westb.seetacloud.com -p 19306
```

- 第一个 -L 是映射vllm的服务到windows上
- 第二个 -L 是映射 mysql的服务到windows上

前面的端口是映射到windows上的端口号，而后面的端口是在AutoDL上启动服务后的端口号

## flask

```
from flask import Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "This is the home page!"
```

**路由将 URL 映射到对应的视图函数的过程，Flask 会根据请求的 URL 来查找对应的视图函数，并执行该函数以生成响应。**

在上面的代码中，  @app.route('/')   表示当用户访问根 URL（即   /  ）时，会调用   home   函数来处理请求，并返回字符串   "This is the home page!"  。

在Flask中，默认只接受GET请求，要接受POST请求需要在路由装饰器中明确指定`methods=['POST']`或`methods=['GET', 'POST']`。**而dify使用flask是进行POST请求**

## 整体代码思路

- 实现功能：

1. 接收用户输入的中文需求并使用 qwen3 转化为 sql 语句。
2. 使用 fask 部署 ap 接口，该 api 用于 执行 sql 语句，并将结果返回,

- 整体代码如下：

```python
from openai import OpenAI
import re
import pymysql
from flask import Flask, jsonify, request

# 创建 Flask 应用
app = Flask(__name__)

def text_2_sql(text):

    prompt = f"""假设你是一个资深的MySQL数据库专家，你能将我给你提交的任务转换成MySQL语句并输出。
请注意：
1. 所有的操作均在名字为student_scores的数据库和student_scores的表上进行。
2. MySQL数据库共有 id、name、chinese、math、english、physics、chemistry、biology、history、geography、politics、total_score共计12列。
样例如下：
​```
用户输入：请帮查找出在名字为王小明的语文成绩是多少？
输出：SELECT chinese FROM student_scores WHERE name = '王小明';
​```
​```
用户输入：请帮查找出英语成绩大于60分的学生名字都有谁？
输出：SELECT name FROM student_scores WHERE english > 60;
​```
​```
用户输入：请帮查找出数学成绩小于90分的学生学号是什么？
输出：SELECT id FROM student_scores WHERE math < 90;
​```
​```
用户输入：请查找出英语成绩大于70分的学生的数学成绩是多少？
输出：SELECT math FROM student_scores WHERE english > 70;
​```
​```
用户输入：{text}。
输出：
​```
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

def get_sql_result(sql):
    # 数据库的操作简易加上异常吃力
    try:
        with connection.cursor() as cursor:
            # 执行sql
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
    finally:
        connection.close()

    return "查询错误"

@app.route("/student", methods=["POST"])
def get_result():

    data = request.get_json()
    natural_language = data.get("text", "").strip()

    if not natural_language:
        return jsonify({"error": "缺少查询内容"}), 400

    # qwen text -> sql
    sql = text_2_sql(natural_language)
    print(sql)
    
    # 执行 sql
    result = get_sql_result(sql) # 结果可以是一个字符 也可以是一个表格

    # 使用 jsonify 包装结果，确保返回合法的 HTTP 响应
    return jsonify({
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
        database='student_scores',      # 要操作的数据库名
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    
    # 应用启动
    app.run(debug=True)
```

执行结果如下：

```sh
PS D:\gx\Desktop\cutting-edge technology\LLM_SQL\src> & D:/python_Develop/anaconda3/envs/py310/python.exe "d:/gx/Desktop/cutting-edge technology/LLM_SQL/src/text2sql.py"
请输入任务要求：请查找出英语成绩大于70分的学生的数学成绩是多少
SELECT 数学 FROM student_scores WHERE 英语 > 70;
```

sql命令执行

```python
def get_sql_result(sql):
    # 数据库的操作简易加上异常吃力
    try:
        with connection.cursor() as cursor:
            # 执行sql
            cursor.execute(sql)
            results = cursor.fetchall()
            return results
    finally:
        connection.close()

    return "查询错误"
```

## POST请求模拟

安装POST-MAN

在 Postman 中正确设置 POST 请求

1. **URL 设置为不带参数的形式** （推荐）

将 URL 改为：

```
http://127.0.0.1:5000/student
```

而不是把自然语言放在路径中。

2. **设置请求方式为 POST**

在 Postman 的下拉菜单中选择 `POST`。

3. **设置 Headers**

点击 `Headers` 标签，查看是否有：Content-Type,application/json，没有就添加

4. **设置 Body**

点击 `Body` 标签，选择 `raw`，然后选择 `JSON` 格式，输入以下内容：

```
{
  "text": "请查找出英语成绩大于70分的学生的数学成绩是多少"
}
```

5. **查看结果**

在下面的Body的JSON中有如下内容显示：

```json
{
    "result": [
        {
            "math": 92
        },
        {
            "math": 89
        },
        {
            "math": 95
        },
        {
            "math": 95
        },
        {
            "math": 82
        },
        {
            "math": 90
        },
        {
            "math": 86
        },
        {
            "math": 93
        },
        {
            "math": 85
        },
        {
            "math": 88
        },
        {
            "math": 91
        },
        {
            "math": 84
        },
        {
            "math": 87
        },
        {
            "math": 94
        },
        {
            "math": 90
        },
        {
            "math": 90
        },
        {
            "math": 86
        },
        {
            "math": 92
        },
        {
            "math": 85
        },
        {
            "math": 91
        }
    ],
    "sql": "SELECT math FROM student_scores WHERE english > 70;"
}
```

## dify连接

进入到dify/docker文件夹执行：docker compose up -d

工作流入下所示：

![](img\1.png)

使用代码执行flask项目

```python
def main(query: str) -> dict:
    import requests
    # 服务地址
    url = "http://host.docker.internal:5000/student"
    data = {
    "text": query
    }
    # 发送 POST 请求
    response = requests.post(url, json=data)

    # 处理响应
    if response.status_code == 200:
        result = response.json()
        return {
            "result": result["result"]
            }
    else:
        return "Man, HaHaHa, What Can I Say? Mamba Out!"
```

1.  data = {"text": query}  里面的键对应flask项目代码的 data.get("text", "").strip()
2.  return { "result": result["result"] } 的键对应 return jsonify({ "result": str(result) })
3.  post请求的连接"http://host.docker.internal:5000/student" 根据自己的情况来，我如此填写是应为我的dify安装在windows的docker里面，不能直接访问windows的网络

运行如下：
![](img\2.png)