import io
import cv2 
import json
import numpy as np 
from PIL import Image  
import triton_python_backend_utils as pb_utils


class TritonPythonModel: 
	def initialize(self, args):
		# You must parse model_config. JSON string is not parsed here
		model_config = json.loads(args["model_config"])

		# Get OUTPUT0 configuration
		output0_config = pb_utils.get_output_config_by_name(
		    model_config, "preprocessing_output"
		)

    	# Convert Triton types to numpy types
		self.output0_dtype = pb_utils.triton_string_to_numpy(
    		output0_config["data_type"]
    	)


	def execute(self, requests):
		"""`execute` MUST be implemented in every Python model. `execute`
    	function receives a list of pb_utils.InferenceRequest as the only
    	argument. This function is called when an inference request is made
    	for this model. Depending on the batching configuration (e.g. Dynamic
    	Batching) used, `requests` may contain multiple requests. Every
    	Python model, must create one pb_utils.InferenceResponse for every
    	pb_utils.InferenceRequest in `requests`. If there is an error, you can
    	set the error argument when creating a pb_utils.InferenceResponse
    	Parameters
    	----------
    	requests : list
    	  A list of pb_utils.InferenceRequest
    	Returns
    	-------
    	list
    	  A list of pb_utils.InferenceResponse. The length of this list must
    	  be the same as `requests`
    	"""

		output0_dtype = self.output0_dtype 

		responses = [] 
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
		for request in requests:

        	# Get INPUT0
			in_0 = pb_utils.get_input_tensor_by_name(
            	request, "preprocessing_input"
          	) 

			image = in_0.as_numpy() 
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = cv2.resize(image, (224, 224))
    
			# Convert the image to float32
			image = image.astype(np.float32)
    
			# Normalize the image (assuming normalization is required)
			# Assuming typical normalization for pretrained models:
			# Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
			mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
			std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
			image = (image / 255.0 - mean) / std
    
			# Transpose the image to convert HWC -> CHW format (required for FORMAT_NCHW)
			image = np.transpose(image, (2, 0, 1))
			image = np.expand_dims(image, axis=0)

			out_tensor_0 = pb_utils.Tensor(
			    "preprocessing_output", image.astype(output0_dtype)
			) 

			inference_response = pb_utils.InferenceResponse(
			    output_tensors=[out_tensor_0]
			)
			responses.append(inference_response) 
		
		return responses


	def finalize(self):
		"""`finalize` is called only once when the model is being unloaded.
		Implementing `finalize` function is OPTIONAL. This function allows
		the model to perform any necessary clean ups before exit.
		"""
		print("Cleaning up...")