<div align="center">
  <a href="https://tommakesthings.github.io/Movie-Genre-Predictor/"><img src="https://github.com/TomMakesThings/Movie-Genre-Predictor/blob/gh-pages/Assets/Readme_Assets/Title.png" width="750"></a>

  <p><b>🎬 Project by <a href="https://github.com/TomMakesThings">TomMakesThings</a>, <a href="https://github.com/rogerchenrc">rogerchenrc</a> and <a href="https://github.com/laviniafr">laviniafr</a></b> 🎬</p>
</div>
  
---
<h1 align="center">About</h1>
  
This is a natural language processing (NLP) group project in which we tested different NLP techniques and model architectures to create a CI/CD pipeline to train and deploy a multi-label classifier. The classifier was trained on dataset of movie descriptions to predict the top fitting genre(s) with 12 possible values including: Drama, Comedy, Action, Crime, Thriller, Romance, Horror, Adventure, Mystery, Family, Fantasy and Sci-Fi. The state of the best trained model was then saved to file and deployed on a custom built web server. For more information, see our <a href="https://tommakesthings.github.io/Movie-Genre-Predictor/">GitHub pages site</a>.
  
<p align="center" href="https://tommakesthings.github.io/Movie-Genre-Predictor/"><img src="https://github.com/TomMakesThings/Movie-Genre-Predictor/blob/gh-pages/Assets/Images/Site-Demo.gif" width="750"></p>
  
<h1 align="center">Runtime Instructions</h1>

**Conda environment:**

To ensure all team members could execute the code during development, it was created using a conda environment. This environment has been saved as a YAML file, environment.yml, and is included in the repository. To recreate this environment:

1. Download the code in the main repository from Code ⮞ Download ZIP
2. Extract the contents of the zip
3. Open the Anaconda prompt and navigate to the folder of the extracted code, e.g. `cd Downloads/Movie-Genre-Predictor`
4. Enter `conda env create -f environment.yml`, where environment.yml is the file path of the enviroment file

**To run the classifier:**
5. From the Anaconda prompt, run `python Web_App/flaskr/main.py` to run the web application

**To run the Jupyter notebooks:**
6. From the Anaconda prompt, run `jupyter notebook`
7. Navigate to the notebook
