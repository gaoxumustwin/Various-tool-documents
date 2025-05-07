import requests
from datetime import datetime, timedelta

def get_weather_forecast(api_key, city):
    """
    获取指定城市未来3天天气预报和昨日历史天气
    
    参数:
        api_key (str): 心知天气API密钥
        city (str): 要查询的城市名称(中文或拼音)
    
    返回:
        dict: 包含天气预报和历史天气的字典
    """
    # API基础URL
    base_url = "https://api.seniverse.com/v3/weather/daily.json"
    
    # 计算日期范围 (昨天 + 今天 + 未来3天)
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    start_date = 0  # 0表示包含昨天
    days = 5        # 总共获取5天数据(昨天1天 + 今天+未来3天)
    
    # 构造请求参数
    params = {
        "key": api_key,
        "location": city,
        "language": "zh-Hans",
        "unit": "c",
        "start": start_date,
        "days": days
    }
    
    try:
        # 发送GET请求
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # 检查请求是否成功
        
        # 解析JSON响应
        weather_data = response.json()
        
        # 提取天气信息
        if "results" in weather_data and weather_data["results"]:
            result = weather_data["results"][0]
            location_info = result["location"]
            daily_data = result["daily"]
            
            # 格式化输出
            weather_info = {
                "城市": location_info["name"],
                "国家": location_info["country"],
                "昨日天气": None,
                "今日天气": None,
                "未来三天预报": []
            }
            
            for day in daily_data:
                date = day["date"]
                day_weather = {
                    "日期": date,
                    "白天天气": day["text_day"],
                    "夜间天气": day["text_night"],
                    "最高温度": f"{day['high']}°C",
                    "最低温度": f"{day['low']}°C",
                    "降水概率": f"{day['rainfall']}%",
                    "风向": day["wind_direction"],
                    "风速": day["wind_speed"]
                }
                
                if date == yesterday.strftime("%Y-%m-%d"):
                    weather_info["昨日天气"] = day_weather
                elif date == today.strftime("%Y-%m-%d"):
                    weather_info["今日天气"] = day_weather
                else:
                    weather_info["未来三天预报"].append(day_weather)
                    
            return weather_info
        else:
            return {"error": "未找到该城市的天气信息"}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"请求天气API时出错: {str(e)}"}
    except ValueError as e:
        return {"error": f"解析天气数据时出错: {str(e)}"}

# 使用示例
if __name__ == "__main__":
    # 替换为你自己的API密钥
    API_KEY = "SzMLSVeMxqGAjanoR"  # 示例密钥，请替换为你的实际密钥
    
    while True:
        city = input("请输入要查询的城市名称(输入q退出): ")
        if city.lower() == 'q':
            break
            
        weather = get_weather_forecast(API_KEY, city)
        
        if "error" in weather:
            print(f"查询失败: {weather['error']}")
        else:
            print(f"\n=== {weather['城市']}天气信息 ===")
            print(f"国家: {weather['国家']}")
            
            # 打印昨日天气
            if weather["昨日天气"]:
                print("\n=== 昨日天气 ===")
                for key, value in weather["昨日天气"].items():
                    print(f"{key}: {value}")
            
            # 打印今日天气
            if weather["今日天气"]:
                print("\n=== 今日天气 ===")
                for key, value in weather["今日天气"].items():
                    print(f"{key}: {value}")
            
            # 打印未来三天预报
            if weather["未来三天预报"]:
                print("\n=== 未来三天天气预报 ===")
                for day in weather["未来三天预报"]:
                    print(f"\n日期: {day['日期']}")
                    print(f"白天: {day['白天天气']}, 夜间: {day['夜间天气']}")
                    print(f"温度: {day['最低温度']} ~ {day['最高温度']}")
                    print(f"降水概率: {day['降水概率']}, 风向: {day['风向']}, 风速: {day['风速']}")
            
            print("\n" + "="*40 + "\n")