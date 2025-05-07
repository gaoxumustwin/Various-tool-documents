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
        family_data = pd.read_csv('../data/home.txt', sep='\t')
        with conn.cursor() as cursor:
            # 先清空表（可选）
            cursor.execute("TRUNCATE TABLE family")
            for _, row in family_data.iterrows():
                cursor.execute(
                    "INSERT INTO family VALUES (%s, %s, %s, %s, %s)",
                    (row['序号'], row['父亲姓名'], row['母亲姓名'], row['父亲电话'], row['母亲电话'])
                )
        
        # 导入身体状况数据
        body_data = pd.read_csv('../data/body.txt', sep='\t')
        with conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE body")
            for _, row in body_data.iterrows():
                cursor.execute(
                    "INSERT INTO body VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (row['序号'], row['身高（cm）'], row['体重（kg）'], row['年龄（岁）'], 
                     row['性别'], row['体脂（%）'], row['血糖（mmol/L）'])
                )

        # 导入成绩数据
        scores_data = pd.read_csv('../data/scores.txt', sep=r'\s+', engine='python')
        with conn.cursor() as cursor:
            # 清空原表数据
            cursor.execute("TRUNCATE TABLE scores")
            # 遍历每一行并插入数据库
            for _, row in scores_data.iterrows():
                cursor.execute(
                    """
                    INSERT INTO scores 
                    (id, name, chinese, math, english, physics, chemistry, biology, history, geography, politics, total_score) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        row['序号'], row['姓名'], row['语文'], row['数学'],
                        row['英语'], row['物理'], row['化学'], row['生物'],
                        row['历史'], row['地理'], row['政治'], row['总分']
                    )
                )

        conn.commit()
        print("数据导入成功")
    except Exception as e:
        conn.rollback()
        print(f"数据导入失败: {e}")
    finally:
        if conn:
            conn.close()

# 执行函数
if __name__ == "__main__":
    import_data()