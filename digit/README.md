# Digit experiment

Extract the digit dataset contained in ./data

To run the code, execute the digit_main.py file. 

Command line parameters are detailed in digit_parser.py.

Results (tf log files) will be stored in ./experiments folder.

Example of usage:
```
# run baseline model
python3 digit_main.py --var '0.020' --lmb 0

# run our model
python3 digit_main.py --var '0.020' --lmb 1
```
