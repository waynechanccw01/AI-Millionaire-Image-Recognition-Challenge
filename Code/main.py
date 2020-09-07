import time
import socketio

from config import endpoint, token, make_directory, parse_question
from model import AI_Model

sio = socketio.Client()
sio.connect("http://%s?token=%s" % (endpoint, token), namespaces=["/contest"])

model = AI_Model()

@sio.on("health-check", namespace="/contest")
def on_health_check():
    return {"ok": True}

@sio.on("question", namespace="/contest")
def on_question(question):
    start = time.time()
    question_id = question["id"]
    question_t = question["type"]
    question_tc = question["typeCode"]
    content = question["content"]

    print(f"Received Question: {question_id}")
    print(f"Question Type: {question_t} ({question_tc})")

    in_1, in_2 = parse_question(content, question_tc)
    ans_ind = model.compare(in_1, in_2, question_tc)

    if question_tc == "M001":
        choices = content["choices"]
        answer = {
            "choiceIds": [choices[_]["id"] for _ in ans_ind]
        }
    elif question_tc == "D001":
        answer = {
            "points": [
                {"x": ans_ind[0][0], "y": ans_ind[0][1]},
                {"x": ans_ind[1][0], "y": ans_ind[1][1]}
            ]
        }
    elif question_tc == "S004":
        answer = {
            "choiceId": ["A", "B"][ans_ind]
        }
    else:
        choices = content["choices"]
        answer = {
            "choiceId": choices[ans_ind]["id"]
        }

    sio.emit("answer", {
        "questionId": question_id,
        "answer": answer
    }, namespace="/contest")


    print(f"answer: {answer}")
    end = time.time()
    time_used = round(end - start, 4)
    print(f'Time used: {time_used}s')
    
    # save question string
    make_directory('../question/')
    now = round(time.time())
    with open(f"../question/q_{now}.txt", 'w') as f:
        f.write(str(question))