from locust import task, between, events, User
import sagemaker
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import json
import os, sys
import time
import numpy as np

endpoint_name=os.environ['ENDPOINT_NAME']
predictor = sagemaker.predictor.Predictor(endpoint_name, 
                                          serializer=JSONSerializer(),
                                          deserializer=JSONDeserializer())
print(predictor.endpoint_name)

csv_test_dir_prefix = 'imdb_data/test'
csv_test_filename = 'test.csv'

# loads a sample and make one inference call
x_test = np.loadtxt(f'{csv_test_dir_prefix}/{csv_test_filename}', 
                    delimiter=',', dtype='int', max_rows=1)
out = predictor.predict(x_test)
print(out)

class SMLoadTestUser(User):
    wait_time = between(0, 1)
    
    @task
    def test_endpoint(self):
        start_time = time.time()
        try:
            predictor.predict(x_test)
            total_time = int((time.time() - start_time) * 1000)
            events.request_success.fire(
                request_type="sagemaker",
                name="predict",
                response_time=total_time,
                response_length=0)

        except:
            total_time = int((time.time() - start_time) * 1000)
            events.request_failure.fire(
                request_type="sagemaker",
                name="predict",
                response_time=total_time,
                response_length=0,
                exception=sys.exc_info())