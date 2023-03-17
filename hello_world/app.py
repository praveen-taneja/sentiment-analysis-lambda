import json

import torch
from transformers import pipeline

model_pipeline = pipeline("sentiment-analysis",
                 model = "/opt/ml/model",
                 tokenizer= "/opt/ml/model")

def lambda_handler(event, context):

    #print(event)

    raw_str = r"{}".format(event['body'])
    body = json.loads(raw_str)
    input_text = body["message"]

    # inputs = tokenizer(input_text, return_tensors="pt")
    # with torch.no_grad():
    #   logits = model(**inputs).logits
    
    # print(f'logits = {logits}')
    # predicted_class_id = logits.argmax().item()
    # predicted_class = model.config.id2label[predicted_class_id]

    result = model_pipeline(input_text)
    #print(result)

    body = {
        "message": "ok",
        "sentiment": result
    }

    response = {
        "statusCode": 200,
        "headers": {
        "Access-Control-Allow-Headers" : "Content-Type",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS"
        },
        "body": json.dumps(body)
    }

    return response

def test_prediction():

    event = {
        "body": '{"message" : "You are nice"}'
    }
    response = lambda_handler(event, None)
    body = json.loads(response["body"])
    print(f"The sentiment is {body['sentiment']}")

    with open('event.json', 'w') as event_file:
        event_file.write(json.dumps(event))

#test_prediction()