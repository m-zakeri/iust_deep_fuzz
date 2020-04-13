# IUST-DeepFuzz
Before getting started please read the documentation:

[IUST-DeepFuzz Website and Documentation](https://m-zakeri.github.io/iust_deep_fuzz/)

## Getting Started
In the current release (0.3.0) you can use IUST-DeepFuzz for test data generation and then fuzzing every application.

### Install
You need to have Python 3.6.x and and up-to-date TensorFlow and Keras frameworks on your computer.
* Install [Python 3.6.x](https://www.python.org/)
* Install [TensorFlow](https://www.tensorflow.org/)
* Install [Keras](https://keras.io/)
* Clone the IUST-DeepFuzz repository: `git clone https://github.com/m-zakeri/iust_deep_fuzz.git` or download the latest version https://github.com/m-zakeri/iust_deep_fuzz.git
* IUST-DeepFuzz is almost ready for test data generation!

### Running
* Configure the `config.py` work with your dataset and to set other paths settings.
* Find the script of specific algorithm that you need. 
* Run the script in command line: `python script_name.py`
* Wait until your file format learn and your test data is generate!

#### Available Pre-trained Models
A pre-trained model is a model that was trained on a large benchmark dataset to solve a problem similar to the one that we want to solve. For the time being, we provided some pre-trained model for *PDF file format*. Our best trained model is available at [model_checkpoint/best_models](model_checkpoint/best_models)

#### Availbale Fuzzing Scripts
ISUT-DeepFuzz has implemented four new deep models and two new fuzz algorithms: DataNeuralFuzz and MetadataNeuralFuzz as our contribution in mentioned thesis. The following algorithms to generate and fuzz test data are available in the current release (r0.3.0):
* `data_neural_fuzz.py`: To implement the DataNeuralFuzz algorithm for fuzzing data in the files.
* `metadata_neural_fuzz.py`: To implement MetadataNeuralFuzz for fuzzing metadata in the files.
* `learn_and_fuzz_3_sample_fuzz.py`: To implement SampleFuzz algorithm introduced in https://arxiv.org/abs/1701.07232. 

#### Available Dataset
Various file format for learning with IUST-DeepFuzz and then fuzz testing is available at [dataset directory](dataset).


## Read More 
[IUST-DeepFuzz Website and Documentation](https://m-zakeri.github.io/iust_deep_fuzz/)


### FAQs

*Last update: April 13, 2020*
