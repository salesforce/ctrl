# LinksTextGeneration

The possibility of having control on the specific generated text is a crucial element in text generation models. In order to achieve this result, we fine-tuned CTRL model by salesforce, which is able to generate conditional text considering several factors such as context, topic or emotion. We trained CTRL model on images' categories of COCO dataset in order to be able to generate context well-related sentences starting from a word, called Control code, that summarizes a concept. Other several specific experiments were hold, to test the accuracy of our model and to better evaluate the choices of CTRL generation’s parameters.
<hr> </hr>

## Usage

In order to perform the fine-tuning of the CTRL model is necessary use tensorflow v1.14, so it is necessary execute the follow command: pip install tensorflow-gpu==1.14

The following commands must be exectued:  
1.     pip install tensorflow-gpu==1.14

2.     patch -b <path_to_tensorflow_estimator_package>/python/estimator/keras.py estimator.patch
3.     pip install fastBPE

It is possible to download two different trained models: one with a sequence length of 256 tokens and another one with 512. 
<br>
In this tutorial, it is used the one with 256 sequence length. It is possible to download it through gsutil.  

4.     pip install gsutil
5.     gsutil -m cp -r gs://sf-ctrl/seqlen256_v1.ckpt . 

Now it is possible to create the TF records, starting from a txt file, and then use it for the training.  

6.     python make_tf_records.py --text_file <path to train_dataset.txt> --control_code <New control code> --sequence_len 256
7.     python training.py --model_dir ../seqlen256_v1.ckpt/ --iterations 256  
The number of iterations depends on the size of training dataset. In order to train for one epoch, it is necessary to take in consideration the batch size.
<br>
Pay attention: the generated tf_record file must be in the directory training_utils.
<br>
In order to generate captions the following command must be exectued:

9.     python generation.py --temperature 0.5 --model seqlen256_v1.ckpt 

<br>
It is possible the default values of generation parameters, like temperature, topK, nucleus and penalty, with the specific flags (have a look generation.py for further details).

10.     python generation.py --temperature 0.5 --nucleus 0.8 --model seqlen256_v1.ckpt 

