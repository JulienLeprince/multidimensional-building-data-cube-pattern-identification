<p align="center">
  <img src="https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/blob/main/figures/README_Header.png" alt="Multidimensional Building Data Cube Mining"/>
</p>

# Multidimensional Building Data Cube Pattern Identification

This work proposes a multi-dimensional analytical method grounded on a generic data mining framework for building data analysis. It puts together reviewed analytics best practices in a step-wise method tailored to DM application for systematic knowledge discovery in big building data. 

Contributions of this work can be summarized as three-fold:
* Framing a multi-dimensional analytical approach leveraging the data-cube structure, cutting down complexity endowed from high-dimensional building data, 
* Putting forward a generic building-tailored data mining framework, 
* Appeal to benchmarking methods for reproducible, comparable & empirically validated analytics.


## Table of content 

-   [Getting Started](#getting-started)
    -   [Data](#source-data)
    -   [Data Analysis](#data-analysis)
<!--    -   [Manuscript and Presentation](#manuscript-and-presentation) -->
-   [Author](#authors)
-   [Licence](#license)

## Getting Started

The directory is divided in sub-folders. Each of which contains the relative source code. Just clone this repository on your computer or Fork it on GitHub

### Source Data

This research uses the open [building-data-genome-project-2](https://github.com/buds-lab/building-data-genome-project-2) data set, available on github. Only the raw data from the BDG2 set are extracted. 
The sets are cleaned and stored under the `data/cleaned/` folder.


### Data Analysis

The source code used for the analysis can be found under the `code` folder, in form of Jupyter notebooks and Python files.

This work follows the [Automated daily pattern filtering of measured building performance data](https://github.com/buds-lab/day-filter) method proposed by Miller et. al, using time series Symbolic Aggregate approXimation (SAX). Daily profiles are segmented, piece-wise approximated and transformed to alphabet sequences representing daily motifs and discords.

The analysis then performs pattern identification as an OLAM process over the 2D lattice of the cube where each cuboids serves for specific insights, i.e., building benchmarking, in-site view and cross building-attribute slice analysis.


<!--### Manuscript and Presentation

The manuscript & presentation of the given work are located under the `manuscript/` folder or can be accessed from the below hyperlinks.

* The manuscript PDF can be found [here](https://github.com/FedericoTartarini/reproducible-research/blob/master/manuscript/presentation_out/presentation.pdf).
* The presentation PDF can be found [here](https://github.com/FedericoTartarini/reproducible-research/blob/master/manuscript/out/main.pdf).
-->



## Authors

* **[Julien Leprince](https://github.com/JulienLeprince)** - *Initial work*
* **[Clayton Miller](https://github.com/cmiller8)** - *Initial work*
* **[Wim Zeiler](https://www.tue.nl/en/research/researchers/wim-zeiler/)** - *Initial work*


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


