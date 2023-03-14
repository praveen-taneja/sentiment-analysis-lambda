import json

import torch
from transformers import pipeline

#from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

#model = joblib.load(trained_model.joblib)

# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")


model_pipeline = pipeline("sentiment-analysis",
                 model = "/opt/ml/model",
                 tokenizer= "/opt/ml/model")

def lambda_handler(event, context):

    #params = event['queryStringParameters']
    #print(f"params = {params}")
    #input_text = params["input_text"]
    print(event)

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
    print(result)

    body = {
        "message": "ok",
        "sentiment": result
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        # "headers": {
        # "Access-Control-Allow-Origin": "*"
        
        # }
    }

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """

def test_prediction():
    # event = {
    #     'queryStringParameters': {
    #     "input_text":"You are nice"
    #     }
    #     }

    event = {
        "body": '{"message" : "You are nice"}'
    }
    response = lambda_handler(event, None)
    body = json.loads(response["body"])
    print(f"The sentiment is {body['sentiment']}")

    with open('event.json', 'w') as event_file:
        event_file.write(json.dumps(event))

#test_prediction()