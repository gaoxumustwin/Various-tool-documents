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