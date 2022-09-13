try:
    from app.database.models.vl_model import VL_Model
except ModuleNotFoundError:
    from vl_model import VL_Model

import requests
import numpy as np
import skimage



class Apimodel(VL_Model):
    '''
    Running APIModel with VL-Interpret:
    python run_app.py -p 6006  -d example_database2  \
                      -m ApiModel
    '''

    def __init__(self):
        self.api_url = "url"
        


    def data_setup(self, example_id: int, image_location: str, input_text: str) -> dict:
        '''
        This method should run a forward pass with your model given the input image and
        text, and return the required data. See app/database/db_example.py for specifications
        of the return data format, and see the implementation in kdvlp.py for an example.
        '''

        return self.dummy_response(example_id, image_location, input_text)

        # to do : adapt to API
        get_data_url = f"{self.api_url}/dummy"

        payload = {
            "img_url": image_location,
            "text": input_text
        }

        response = requests.post(get_data_url, data=payload)
        parsed_response = response.json()

        len_img = len(parsed_response["image"])
        txt_tokens = parsed_response["tokens"] 
        hidden_state = parsed_response["hidden"]

        return {
            'ex_id': example_id,
            'image': parsed_response["image"],
            'tokens': parsed_response["tokens"],
            'txt_len': len(txt_tokens),
            'attention': np.array(parsed_response["attention"]),
            'img_coords': parsed_response["img_coords"],
            'hidden_states': np.array(hidden_state)
        }
    



    def dummy_response(self,example_id,  image_location, input_text):

        # Retriev the image
        response = requests.get(image_location)
        image_numpy = skimage.io.imread( image_location )

        splitted = input_text.split(" ")
        txt_len = len(splitted)+2
        tokens = ["[CLS]"] + splitted + ['[SEP]','IMG_0', 'IMG_1', 'IMG_2', 'IMG_3', 'IMG_4', 'IMG_5']


        return {
            'ex_id': example_id,
            'image': image_numpy,

            'tokens': tokens,

            'txt_len': txt_len,
            'img_coords': [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
            'accuracy': example_id % 2 == 0,  

            'attention': np.random.rand(12, 12, 14, 14),
            'hidden_states': np.random.rand(13, 50, 768),
            'custom_metrics': {'Example Custom Metrics': np.random.rand(12, 12)}
        }
