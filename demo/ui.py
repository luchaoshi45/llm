from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

app = FastAPI()
bot = QABot(model_path=_LLM_, device="cuda")

@app.get("/", response_class=HTMLResponse)
async def chat_page():
    return """
    <form action="/ask" method="post">
        <input type="text" name="question" placeholder="请输入问题">
        <button type="submit">提问</button>
    </form>
    """

@app.post("/ask")
async def ask_question(request: Request):
    form_data = await request.form()
    question = form_data["question"]
    answer = bot.ask(question)
    return {"question": question, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)