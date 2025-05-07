import json
import requests
from datetime import datetime
from typing import Dict, Callable
from openai import OpenAI

# 1. 定义真实天气查询函数（替换原有模拟函数）
def get_current_temperature(location: str) -> Dict:
    """使用心知天气API获取指定位置的当前温度"""
    api_key = "SzMLSVeMxqGAjanoR"  # 替换为你的实际API密钥
    base_url = "https://api.seniverse.com/v3/weather/now.json"
    
    params = {
        "key": api_key,
        "location": location,
        "language": "zh-Hans",
        "unit": "c"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "results" in data and data["results"]:
            result = data["results"][0]
            return {
                "temperature": float(result["now"]["temperature"]),
                "location": location,
                "unit": "celsius",
                "weather": result["now"]["text"],
                "last_update": result["last_update"]
            }
        else:
            return {"error": "未找到该城市的天气信息"}
    except Exception as e:
        return {"error": f"查询天气API时出错: {str(e)}"}

def get_temperature_date(location: str, date: str) -> Dict:
    """使用心知天气API获取指定位置和日期的预测温度"""
    api_key = "SzMLSVeMxqGAjanoR"  # 替换为你的实际API密钥
    base_url = "https://api.seniverse.com/v3/weather/daily.json"
    
    # 计算日期差
    target_date = datetime.strptime(date, "%Y-%m-%d")
    today = datetime.now()
    days_diff = (target_date - today).days + 1  # +1因为API的start=0表示包含今天
    
    if days_diff < 0:
        return {"error": "日期不能早于今天"}
    if days_diff > 3:
        return {"error": "免费用户只能查询未来3天的预报"}
    
    params = {
        "key": api_key,
        "location": location,
        "language": "zh-Hans",
        "unit": "c",
        "start": days_diff,
        "days": 1
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "results" in data and data["results"]:
            result = data["results"][0]
            daily = result["daily"][0]
            return {
                "temperature": (float(daily["low"]) + float(daily["high"])) / 2,  # 取平均温度
                "location": location,
                "date": date,
                "unit": "celsius",
                "day_weather": daily["text_day"],
                "night_weather": daily["text_night"],
                "high": float(daily["high"]),
                "low": float(daily["low"])
            }
        else:
            return {"error": "未找到该城市的天气信息"}
    except Exception as e:
        return {"error": f"查询天气API时出错: {str(e)}"}

# 工具注册表
FUNCTION_REGISTRY: Dict[str, Callable] = {
    "get_current_temperature": get_current_temperature,
    "get_temperature_date": get_temperature_date
}

def get_function_by_name(name: str) -> Callable:
    """根据名称获取函数"""
    return FUNCTION_REGISTRY[name]

# 2. 定义工具描述（供AI理解可用工具）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "获取指定位置的当前温度（使用心知天气API）",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市和地区，例如'北京'或'beijing'",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "获取指定位置和日期的预测温度（使用心知天气API）",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市和地区，例如'北京'或'beijing'",
                    },
                    "date": {
                        "type": "string",
                        "description": "ISO 8601格式的日期，例如'2024-10-01'",
                    },
                },
                "required": ["location", "date"],
            },
        },
    },
]

# 3. 初始化客户端
openai_api_key = "EMPTY"
openai_api_base = "http://127.0.0.1:8080/v1"


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model_name = "Qwen3-8B"

# 4. 初始化对话
messages = [
    {
        "role": "system",
        "content": "你是一个天气助手，可以使用心知天气API查询实时天气和预报。当前日期: " + datetime.now().strftime("%Y-%m-%d")
    },
    {
        "role": "user",
        "content": "北京现在的温度是多少？明天会下雨吗？"
    }
]

# 5. 第一轮请求 - 获取工具调用
print("=== 第一轮请求: 获取工具调用 ===")
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=TOOLS,
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    presence_penalty=1.5,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False}, # 关闭思考模式
        "repetition_penalty": 1.05,
    },
)


# 第一轮响应处理
assistant_message = response.choices[0].message
messages.append(assistant_message.model_dump())  # 正确添加到消息历史

print("\n=== AI返回的工具调用 ===")
if assistant_message.tool_calls:
    tool_calls = [call.model_dump() for call in assistant_message.tool_calls]
    print(json.dumps(tool_calls, indent=2, ensure_ascii=False))
else:
    print("没有调用任何工具")

# === AI返回的工具调用 ===
# [
#   {
#     "id": "chatcmpl-tool-e67cc26c04cc415ebcb03dee3a497841",
#     "function": {
#       "arguments": "{\"location\": \"北京\"}",
#       "name": "get_current_temperature"
#     },
#     "type": "function"
#   },
#   {
#     "type": "function"
#   }
# ]


if tool_calls := assistant_message.tool_calls:
    print("\n=== 执行工具调用 ===")
    for tool_call in tool_calls:
        call_id = tool_call.id
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)
        
        print(f"调用工具: {fn_name}, 参数: {fn_args}")
        
        # 执行函数并获取结果
        fn_res = get_function_by_name(fn_name)(**fn_args)
        
        # 处理API错误
        if "error" in fn_res:
            fn_res = {"error": fn_res["error"]}
        
        # 将结果添加到消息历史
        messages.append({
            "role": "tool",
            "content": json.dumps(fn_res, ensure_ascii=False),
            "tool_call_id": call_id,
        })

# === 执行工具调用 ===
# 调用工具: get_current_temperature, 参数: {'location': '北京'}
# 调用工具: get_temperature_date, 参数: {'location': '北京', 'date': '2025-05-08'}

# 7. 第二轮请求 - 获取最终回答
print("\n=== 第二轮请求: 获取最终回答 ===")
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    tools=TOOLS,
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    presence_penalty=1.5,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False}, # 关闭思考模式
        "repetition_penalty": 1.05,
    },
)

# 8. 处理最终回答
final_message = response.choices[0].message
messages.append(final_message.model_dump())

print("\n=== 最终回答 ===")
print(final_message.content)

# === 最终回答 ===
# 北京现在的温度是23摄氏度，天气晴朗，数据更新于2025年5月7日14:40。

# 明天（2025年5月8日），北京的气温范围将在14到25摄氏度之间，白天和夜晚都会有雷阵雨。建议外出时携带雨具并注意防雷电安全。

print("\n=== 完整的消息历史 ===")
print(json.dumps(messages, indent=2, ensure_ascii=False))

# === 完整的消息历史 ===
# [
#   {
#     "role": "system",
#     "content": "你是一个天气助手，可以使用心知天气API查询实时天气和预报。当前日期: 2025-05-07"
#   },
#   {
#     "role": "user",
#     "content": "北京现在的温度是多少？明天会下雨吗？"
#   },
#   {
#     "content": null,
#     "refusal": null,
#     "role": "assistant",
#     "annotations": null,
#     "audio": null,
#     "function_call": null,
#     "tool_calls": [
#       {
#         "id": "chatcmpl-tool-e67cc26c04cc415ebcb03dee3a497841",
#         "function": {
#           "arguments": "{\"location\": \"北京\"}",
#           "name": "get_current_temperature"
#         },
#         "type": "function"
#       },
#       {
#         "id": "chatcmpl-tool-cb60b5757ba04d5d8e907db51c8ba049",
#         "function": {
#           "arguments": "{\"location\": \"北京\", \"date\": \"2025-05-08\"}",
#           "name": "get_temperature_date"
#         },
#         "type": "function"
#       }
#     ],
#     "reasoning_content": null
#   },
#   {
#     "role": "tool",
#     "content": "{\"temperature\": 23.0, \"location\": \"北京\", \"unit\": \"celsius\", \"weather\": \"晴\", \"last_update\": \"2025-05-07T14:40:19+08:00\"}",
#     "tool_call_id": "chatcmpl-tool-e67cc26c04cc415ebcb03dee3a497841"
#   },
#   {
#     "role": "tool",
#     "content": "{\"temperature\": 19.5, \"location\": \"北京\", \"date\": \"2025-05-08\", \"unit\": \"celsius\", \"day_weather\": \"雷阵雨\", \"night_weather\": \"雷阵雨\", \"high\": 25.0, \"low\": 14.0}",
#     "tool_call_id": "chatcmpl-tool-cb60b5757ba04d5d8e907db51c8ba049"
#   },
#   {
#     "content": "北京现在的温度是23摄氏度，天气晴朗，数据更新于2025年5月7日14:40。\n\n明天（2025年5月8日），北京的气温范围将在14到
# 25摄氏度之间，白天和夜晚都会有雷阵雨。建议外出时携带雨具并注意防雷电安全。",
#     "refusal": null,
#     "role": "assistant",
#     "annotations": null,
#     "audio": null,
#     "function_call": null,
#     "tool_calls": [],
#     "reasoning_content": null
#   }
# ]