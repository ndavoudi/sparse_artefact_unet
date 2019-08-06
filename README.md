# In-vivo optoacoustic sparse reconstruction artefact removal with U-Net 

This repository contains the code of sparse reconstruction artefact removal with convolutional neural network that was employed in our work:

## Requirements


## Running the code

### Training


### Testing

You can downloed a trained model by running the script "download_pretrained_model_32.sh" 
```
sh download_pretrained_model_32.sh
```
which will add the model to the created directory.
Then by running "test.py" 
```
python test.py
```
you will use the downloded trained model and provided sample test data to test the network. Sample test data includes the
network input as artefactual sparse recostruction images, "test_32.mat", and ground truth artefact-free full reconstruction, "test_GT.mat", to be compared with network 
output for performance evaluation.

## Some results from our paper
