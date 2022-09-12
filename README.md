# IUST-DeepFuzz

Before getting started, please read the documentation:

[IUST-DeepFuzz Website and Documentation](https://m-zakeri.github.io/iust_deep_fuzz/)

and watch the DeepFuzz demo:

[Video demo](http://parsa.iust.ac.ir/wp-content/uploads/2021/06/IUST-DeepFuzz2020_Demo.mp4)


## Getting Started
In the current release (0.3.0), you can use IUST-DeepFuzz for test data generation and then fuzz every application.

### Install
You need Python 3.6.x and up-to-date TensorFlow and Keras frameworks on your computer.
* Install [Python 3.6.x](https://www.python.org/)
* Install [TensorFlow](https://www.tensorflow.org/)
* Install [Keras](https://keras.io/)
* Clone the IUST-DeepFuzz repository: `git clone https://github.com/m-zakeri/iust_deep_fuzz.git` or download the latest version https://github.com/m-zakeri/iust_deep_fuzz.git
* IUST-DeepFuzz is almost ready for test data generation!

### Running
* Configure the `config.py` work with your dataset and set other path settings.
* Find the script of the specific algorithm that you need. 
* Run the script in the command line: `python script_name.py`
* Wait until your file format learns and your test data is generated!

#### Available Pre-trained Models
A pre-trained model is a model that was trained on a large benchmark dataset to solve a problem similar to the one that we want to solve. For the time being, we provided some pre-trained models for *PDF file format*. Our best trained model is available at [model_checkpoint/best_models](model_checkpoint/best_models)

#### Available Fuzzing Scripts
ISUT-DeepFuzz has implemented four new deep models and two new fuzz algorithms: DataNeuralFuzz and MetadataNeuralFuzz, as our contributions of the mentioned thesis. The following algorithms to generate and fuzz test data are available in the current release (r0.3.0):

* `data_neural_fuzz.py`: To implement the DataNeuralFuzz algorithm for fuzzing data in the files.
* `metadata_neural_fuzz.py`: To implement MetadataNeuralFuzz for fuzzing metadata in the files.
* `learn_and_fuzz_3_sample_fuzz.py`: To implement the SampleFuzz algorithm introduced in https://arxiv.org/abs/1701.07232. 

#### Available Dataset
Various file format for learning with IUST-DeepFuzz and then fuzz testing is available at [dataset directory](dataset).


## Read More 
Recently, I wrote a blog post about our DeepFuzz paper:

* [Innovations on Automatic Test Data Generation](https://m-zakeri.github.io/innovations-on-automatic-test-data-generation.html#innovations-on-automatic-test-data-generation)


### FAQs
if you have any questions, please do not hesitate to contact me:

[m-zakeri@live.com](mailto:m-zakeri@live.com)

Last update: **September 12, 2022**

