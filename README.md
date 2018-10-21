[//]: # (Image References)
[image1]: https://raw.githubusercontent.com/mbabin2/P3-Collaboration-Competition/master/images/anaconda.png "Conda"
[image2]: https://raw.githubusercontent.com/mbabin2/P3-Collaboration-Competition/master/images/jupyter_home-collab.png "Home"
[image3]: https://raw.githubusercontent.com/mbabin2/P3-Collaboration-Competition/master/images/set_kernal.png "Kernel"

# Udacity "DRLN" - Project 3 Submission

## 1. Environment Details

For this project, we were tasked with training two agents to cooperatively hit a ball back and forth over a net. The reward function for the environment has the agents gain +0.1 for hitting the ball over the net, and -0.01 if they let the ball hit the floor, hit it off the table completely. Thus the goal of this task was to have both agents learn to volley the ball back and forth for as long as possible. This task requires the agents to average a score of +0.50 over 100 consecutive episodes in order to be considered solved.

The state space for this environment contains 8 dimensions, and only 2 continuous actions the agent can take.

## 2. Dependencies

Inorder to run this project, please follow the instructions below:

1. Install [Anaconda](https://www.anaconda.com/download/#windows).
![Conda][image1]


2. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```


3. Clone the repository.
```bash
git clone https://github.com/mbabin2/P1-Navigation
```


4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```


5. Open the Jupyter Notebook using the following command from the `p3_collab-compet/` folder.
```bash
jupyter notebook
```


6. Select either `Tennis-Test.ipynb` to view a pre-trained agent, or `Tennis-Train.ipynb` to train an agent for yourself.
![Home][image2]


7. Before running any code in a notebook, make sure to change the kernel to `drlnd` under the `Kernel` menu and `Change kernel` sub-menu. 

![Kernel][image3]

## 3. The Jupyter Notebooks

Any instructions for executing the code within the Jupyter Notebooks will be included inside the notebooks themselves. Any explanations for the implementation details of algorithms will be outlined in the `Report.pdf` file found at the root of this repository.
