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

    