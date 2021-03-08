[![MIT License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/sysgenerated/CreditScoring">
    <img src="images/github.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Credit Scoring</h3>

  <p align="center">
    A training and scoring experiment.
    <br />
    <br />
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <br>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#r-markdown">R Markdown Exploration</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://github.com/sysgenerated/CreditScoring)

This repository provides training and predicting python scripts for a machine learning pipeline.

### Built With


* [Pycharm](https://www.jetbrains.com/pycharm/)
* [RStudio](https://rstudio.com/)
* [scikit-learn](https://scikit-learn.org/stable/index.html)
* [XGBoost](https://xgboost.readthedocs.io/en/latest/index.html)
* [LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html)


<!-- R Markdown -->
## R Markdown

An R markdown notebook with exploratory analysis is rendered at the following link:

<a href="https://sysgenerated.github.io/CreditScoring/">R Markdown</a>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

Requirements can be installed via pip or conda.
* pip
  ```sh
  pip install -r requirements.txt
  ```

* conda
  ```sh
  conda install --file requirements.txt
  ```


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/sysgenerated/CreditScoring.git
   ```



<!-- USAGE EXAMPLES -->
## Usage

Scoring
1. Add an input file to the project root folder
2. Run prediction.py
   ```sh
   python prediction.py input_data.csv 
   ```

Scoring Help
1. Help is also available
   ```sh
   python prediction.py -h 
   ```

Training
1. Add an input file to the project root folder
2. Run training.py
   ```sh
   python training.py input_data.csv 
   ```

Training Help
1. Help is also available
   ```sh
   python training.py -h 
   ```


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/sysgenerated/CreditScoring/blob/master/LICENSE.txt
[product-screenshot]: images/credit_score.png
