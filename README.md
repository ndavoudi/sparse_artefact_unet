# In-vivo optoacoustic sparse reconstruction artefact removal with U-Net 

This repository contains the code of sparse reconstruction artefact removal with convolutional neural network that was employed in our work: [Deep learning optoacoustic tomography with sparse data](https://www.nature.com/articles/s42256-019-0095-3)


![Screenshot 2024-07-03 at 14 48 51](https://github.com/ndavoudi/sparse_artefact_unet/assets/53782756/8064c7a5-7c5d-495d-908d-ac3f9f578384)


## Running the code


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


## Citation
Please cite the following paper if you use this code:

Davoudi, Neda, Xosé Luís Deán-Ben, and Daniel Razansky. "Deep learning optoacoustic tomography with sparse data." Nature Machine Intelligence 1.10 (2019): 453-460.
