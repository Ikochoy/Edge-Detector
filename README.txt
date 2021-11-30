** Folder Structure **
assignment1/
├── A1_images        # a directory that contains raw images
|   ├── image1.jpg   # 1st image provided
|   ├── image2.jpg   # 2nd image provided
|   ├── jiufen.jpeg  # self selected image
├── output_images    # a directory that output images are saved
├── a1_functions.py  # python file that contains all the edge detector codes
├── a1Test&Visualize.ipynb # jupyter notebook that visualizes the images and
|                            contains testing code
├── requirements.txt # a txt file that contains all the packages needed
├── README.txt       # this txt file
├── report.pdf       # the report pdf file with all my solutions

a1_functions.py -- contains all the code for each implementation task (q1 to q3)
a1Test&Visualize.ipynb -- a notebook that contains visualization task for q1
and all the codes needed for testing (q4)


** How to run my code **

Suggested steps for running my functions:

1) Make sure you cd into the assignment1 directory

2) Create a virtual environment using the command below. Note that python
version that I am using is python3.7
```
conda create -n name python=3.7
```

3) Before running the code, activate your environment and make sure that the environment contains all the necessary packages used for computations and visualization.
Install all libraries/packages with the provided `requirements.txt`
```
pip install -r requirements.txt
```

4) To see the tests, and see how we can apply different steps of the edge
detector of the images, one can open up `a1Test&Visualize.ipynb` using the
```jupyter notebook``` command in the terminal and run the codes. There are
comments in the notebook that explains what each important line of code is doing.
> NOTE: If an Import Error (ImportError: The Jupyter Server requires tornado >=6.1.0) is found when ```jupyter notebook``` is typed in terminal, run ```pip install tornado --upgrade``` and ```jupyter notebook``` again. 


** To run the edge detector in new images **
1) Add the image into A1_images/
2) open the `a1Test&Visualize.ipynb` file and navigate to the last section
'Template Code chunk for running edge detector on new image'
3) Enter the name of your newly selected image. Note that the name should also
include the file format, e.g: "new_image.jpg"
4) Run the first two block (that contain the imports statements)
5) Run the last code block (the template code chunk section)
6) Unless you change the path of saved images, otherwise output images would be
found in the output_images directory by default


** Report.pdf **
All the visualization results in the report.pdf can be found in the output
directory
