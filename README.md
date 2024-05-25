## Requirements 

```
python -m pip install -r requirements.txt
```
This code was tested with python 3.8 and torch 2.0.0. 

## Repository Structure 
```
├── ddm_inversion - folder contains inversions in order to work on real images: ddim inversion as well as ddpm inversion (our method).
├── example_images - folder of input images to be edited
├── imgs - images used in this repository readme.md file
├── prompt_to_prompt - p2p code
├── main_run.py - main python file for real image editing
└── test.yaml - yaml file contains images and prompts to test on
```

A folder named 'results' will be automatically created and all the results will be saved to this folder. We also add a timestamp to the saved images in this folder.

## Algorithm Inputs and Parameters
Method's inputs: 
```
init_img - the path to the input images
source_prompt - a prompt describing the input image
target_prompts - the edit prompt (creates several images if multiple prompts are given)
```
These three inputs are supplied through a YAML file (please use the provided 'test.yaml' file as a reference).

<br>
Method's parameters are:

```
skip - controlling the adherence to the input image
cfg_tar - classifier free guidance strengths
```
These two parameters have default values, as descibed in the paper.

## Usage Example 
```
python3 main_run.py --mode="our_inv" --dataset_yaml="test.yaml" --skip=36 --cfg_tar=15 
python3 main_run.py --mode="p2pinv" --dataset_yaml="test.yaml" --skip=12 --cfg_tar=9 

```
The ```mode``` argument can also be: ```ddim``` or ```p2p```.

In ```our_inv``` and ```p2pinv``` modes we suggest to play around with ```skip``` in the range [0,40] and ```cfg_tar``` in the range [7,18].

**Controlnet and Cross frames self-attention**:
```
python3 main_run_my_c_cf_videos.py
```
tips: You need to replace package diffusers/models/attention_processor.py with diffusers_cf_attn/attention_processor.py