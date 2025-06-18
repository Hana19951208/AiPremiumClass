import os
from dotenv import load_dotenv, find_dotenv
from zhipuai import ZhipuAI

if __name__ == "__main__":

    load_dotenv(find_dotenv())
    # 创建调用客户端
    client = ZhipuAI(
        api_key=os.environ["api_key"],
    )
    response = client.chat.completions.create(
        model="glm-4-flash-250414",
        messages=[
            {
                "role": "user",
                "content": "天空是啥样的？",
            }
        ],
        top_p=0.2,
        temperature=0.5
    )
    print(response.choices[0].message.content)
