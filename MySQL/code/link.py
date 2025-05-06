import pymysql

# 连接配置
connection = pymysql.connect(
    host='127.0.0.1',        # 本地地址
    # port=3306,               # WSL端口
    port=6006,               # Autodl端口
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