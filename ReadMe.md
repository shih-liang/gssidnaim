# Code for "Global spatiotemporal synchronizing structures of spontaneous neural activities in different cell types"

## Dependencies
 * Python.
 * Python packages in "requirements.txt".

## How to run it

* Install Python.
* Install all required Python packages using something like ```pip install -r requirements.txt```.
* Download ```.npy``` files on [our data sharing repo](http://eai.brainsmatics.org/datasharing/shi2402) and place the files you want to process in ```demodata``` directory.
* You can just run ```python demo.py``` to get a demo result. Or if you prefer Jupyter, simply open ```demo.ipynb```. You may need to modify the data path or file names in the code to successfully run it.

Please refer to ```demo.ipynb``` for instructions on how to execute the code on your data and to view the expected outputs. Please note that lines 24 to 26 in ```demo.py``` may take a few minutes to execute, so please be patient.

## Tested on
  * Arch Linux (6.4.10-arch1-1).
  * Python 3.11.3.
  * At least 128G memory if you want compute FC of all pixel pairs (using ```rsfc.full_image_pipeline``` function).
