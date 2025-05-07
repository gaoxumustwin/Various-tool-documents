import requests

def get_weather(api_key, city):
    """
    查询指定城市的当前天气情况
    
    参数:
        api_key (str): 心知天气API密钥
        city (str): 要查询的城市名称(中文或拼音)
    
    返回:
        dict: 包含天气信息的字典
    """
    # API基础URL
    base_url = "https://api.seniverse.com/v3/weather/now.json"
    
    # 构造请求参数
    params = {
        "key": api_key,
        "location": city,
        "language": "zh-Hans",
        "unit": "c"
    }
    
    try:
        # 发送GET请求
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # 检查请求是否成功
        
        # 解析JSON响应
        weather_data = response.json()
        
        # 提取主要天气信息
        if "results" in weather_data and weather_data["results"]:
            result = weather_data["results"][0]
            return {
                "城市": result["location"]["name"],
                "天气状况": result["now"]["text"],
                "温度": f"{result['now']['temperature']}°C",
                "更新时间": result["last_update"]
            }
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
            
        weather = get_weather(API_KEY, city)
        
        if "error" in weather:
            print(f"查询失败: {weather['error']}")
        else:
            print("\n=== 当前天气信息 ===")
            print(f"城市: {weather['城市']}")
            print(f"天气: {weather['天气状况']}")
            print(f"温度: {weather['温度']}")
            print(f"更新时间: {weather['更新时间']}\n")