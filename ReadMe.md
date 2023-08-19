# Code for "Global spatiotemporal structure and its deviations of neural activities in mice: a resting-state waves perspective"

## Dependencies
 * Python.
 * Python packages in "requirements.txt".

## How to run it

* Install Python.
* Install all required Python packages using something like ```pip install -r requirements.txt```.
* Download all ```.npy``` files on [huggingface](https://huggingface.co/datasets/iliang/NeuralActivityDemo/tree/main/demodata) and place them in ```demodata``` directory. These files are sample data.
* You can just run ```python demo.py``` to get a demo result. Or if you prefer jupyter, simply open ```demo.ipynb```.

Please refer to ```demo.ipynb``` for instructions on how to execute the code on your data and to view the expected outputs. Please note that lines 24 to 26 in ```demo.py``` may take a few minutes to execute, so please be patient.

## Tested on
  * Arch Linux (6.4.10-arch1-1).
  * Python 3.11.3.
  * At least 128G memory if you want compute FC of all pixel pairs (using ```rsfc.full_image_pipeline``` function).
