# from openai import OpenAI
# import re

# def text_2_sql(text):

#     prompt = f"""假设你是一个资深的MySQL数据库专家，你能将我给你提交的任务转换成MySQL语句并输出。
# 请注意：
# 1. 所有的操作均在名字为student_scores的数据库和student_scores的表上进行。
# 样例如下：
# ```
# 用户输入：请帮查找出在名字为王小明的语文成绩。
# 输出：SELECT 语文 FROM student_scores WHERE 姓名 = '王小明';
# ```
# ```
# 用户输入：请帮查找出英语成绩大于60分的学生名字。
# 输出：SELECT 姓名 FROM student_scores WHERE 英语 > 60;
# ```
# ```
# 用户输入：{text}。
# 输出：
# ```
# """

#     chat_response = client.chat.completions.create(
#         model="Qwen3-0.6B",
#         temperature=0.5,
#         top_p=0.7,
#         presence_penalty=1.5,
#     	extra_body={"chat_template_kwargs": {"enable_thinking": False}},
#         messages=[
#             {"role": "system", "content": "你是一个数据库助手"},
#             {"role": "user", "content": prompt},
#         ],
#     )
#     sql = chat_response.choices[0].message.content

#     # 使用正则表达式提取 SQL 语句
#     match = re.search(r'```(?:sql)?\n(.*?)\n```', sql, re.DOTALL)

#     if match:
#         sql_query = match.group(1).strip()

#         return sql_query
    
#     else:
#         return sql



# if __name__ == "__main__":

#     openai_api_key = "EMPTY"
#     openai_api_base = "http://127.0.0.1:8000/v1"
#     client = OpenAI(
#         base_url=openai_api_base,
#         api_key=openai_api_key
#     )

#     text = input("请输入任务要求：")

#     print(text_2_sql(text))



def main(query: str) -> dict:
    import requests
    # 服务地址
    url = "http://192.168.2.101:8010/text2sql"
    data = {
    "query": query
    }
    # 发送 POST 请求
    response = requests.post(url, json=data)

    # 处理响应
    if response.status_code == 200:
        result = response.json()
        return {
            "result": result["sql_result"]
            }
    else:
        return "Man, HaHaHa, What Can I Say? Mamba Out!"