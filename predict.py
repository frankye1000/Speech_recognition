import os
import pickle
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\金鑰位置.json' #金鑰驗證
from google.api_core.client_options import ClientOptions
from google.cloud import automl_v1

# 讀取文字檔
def inline_text_payload(file_path):
  with open(file_path, 'rb') as ff:
    content = ff.read()
    print(content)
  return {'text_snippet': {'content': content, 'mime_type': 'text/plain'} }

# 讀取PDF檔案
def pdf_payload(file_path):
  return {'document': {'input_config': {'gcs_source': {'input_uris': [file_path] } } } }

# 預測
def get_prediction(model_name, content):
  options = ClientOptions(api_endpoint='automl.googleapis.com')
  prediction_client = automl_v1.PredictionServiceClient(client_options=options)

  payload = {'text_snippet': {'content': content, 'mime_type': 'text/plain'} }
  params = {}
  request = prediction_client.predict(model_name, payload, params)

  return request  # waits until request is returned

if __name__ == '__main__':
  model_name = '模型位置'

  with open('test_rowdata.pickle', 'rb') as file:
    test_rowdata = pickle.load(file)
  for content in test_rowdata:
    pred_return = get_prediction(model_name,content)
    print(pred_return.payload[0].display_name)