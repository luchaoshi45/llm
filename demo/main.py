from bot import QABot

if __name__ == "__main__":
    # 初始化机器人
    bot = QABot(model_path=_LLM_, device="cuda")

    # 示例：问答功能
    while True:
        question = input("你问: ")
        if question.lower() in ["退出", "exit", "quit"]:
            print("机器人: 再见！")
            break
        answer = bot.ask(question)
        print(f"机器人: {answer}")

    # 示例：评估功能
    print("正在评估模型在 ARC Challenge 上的表现...")
    results = bot.evaluate("arc_challenge")
    print(results)