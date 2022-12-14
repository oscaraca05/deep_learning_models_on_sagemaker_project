# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Remember that your README should:
- Include a screenshot of completed training jobs
![alt text](hyperparameter_tuning.PNG)
- Logs metrics during the training process
![alt text](log_metrics_training.PNG)
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs
![alt text](best_model_hyperparameters.PNG)

## Debugging and Profiling
First, it is necessary to configure the rules for both debugging and profiling. Then you have to configure the hooks for the training and test phase, you also have to attach the loss function to the hook. With the correct configurations all the debugging and profiling outputs will reside within s3 training job directory.

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

![alt text](loss_training_validation.PNG)
Answer: First, we can see how the loss keeps decreasing throghout time, but we also can see a lower slope at the end, it suggests that we are reaching the necessary epochs for the model to converge. If we had set more epochs, we may had realize that it is not necesary to continue the training phase after the loss convergence.

**TODO** Remember to provide the profiler html/pdf file in your submission.

Answer: You can find the pdf profiler report in the file "profiler-report.html" at the root of the project.
For the first trials I did with the profiler, I could see in the Profiler Report a lot of graphs showing memory and cpu usage, training and evaluation time spent, among other interesting stuff. I remember that the CPU usage was like 100% all the time but the memory usage was below 15%, with this info we may setup a cheaper host and get similar results in terms of time-performance. In the later trials, these graphs disappeared from the report but the table with the rules summary remains, I don't know why, I would appreciate feedback about that. It was a pretty challenging project and I would like to take the most of it.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
Answer: I wrote an isolated inference.py script for deploying the model, within this script I configured the function input_fn which allows the user asking for inferences only passing an image's bytes stream. The functions take care of the image by transforming it to a tensor and also resizing it to match with the model's input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
![alt text](pytorch_working_endpoint.PNG)

## Standout Suggestions
**TODO (Optional):** This is where you can provide information about any standout suggestions that you have attempted.
