<p align="center">
  <img src="https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/blob/main/figures/README_Header.png" alt="Multidimensional Building Data Cube Mining"/>
</p>

# Multidimensional Building Data Cube Pattern Identification

This work presents the implementation section of the journal article [Data mining cubes for buildings, a generic framework for multidimensional analytics of building performance data](https://doi.org/10.1016/j.enbuild.2021.111195).
The mentioned article puts forward a multidimensional method leveraging the well-established structures of data cubes to tackle the complexities endowed from high-dimensional building data.

In short, the work compels building analysts to a common data mining approach from the four following contributions:
1.	Establishing a generic building-analysis data mining framework for unified, systematic and more interpretable analytics.  
2.	Framing a multidimensional approach to big building data from conclusively adopted data cube structures, while linking dimensional frames to application- and insight-driven approaches.
3.	Assembling and/or employing open building data test sets to serve as benchmarks for studies, and delivering more comparable & empirically validated analytics. In this case, the Building Data Genome Project 2 was used, serving as an open test-case illustration comprising 3,053 energy meters from 1,636 buildings.
4.	Finally, developing replicable and open-source implementations of typical building energy management applications, thus enhancing knowledge transfer within the field; which we endorsed by providing open access implementations of our applied method, here presented.

We believe ensuing these steps will cultivate more generalizable findings and insights while vastly contributing to the practical adoption of a common language for data analytics within the building sector. 

This repository comprises the automated pattern filtering application, [dayfilter](https://github.com/buds-lab/day-filter), from the work of Miller et. al (2015) extended to a 3-dimensional building-cube, revealing three typical building-analytical approaches, i.e., top-down, bottom-up and temporal drill-in. 

Discovered knowledge is illustrated through impactful and effective visualizations to allow human inspection while paving the way towards a wider adoption of more interpretable mining processes. Our findings highlight the importance of a priori defining dimensional spaces to explore through mining, driven by explicit questions the analyst is seeking to answer, e.g., “How does my building perform compared to its peers?”.


## Table of content 

-   [Getting Started](#getting-started)
    -   [Data](#source-data)
    -   [Data Analysis](#data-analysis)
<!--    -   [Manuscript and Presentation](#manuscript-and-presentation) -->
-   [Author](#authors)
-   [Licence](#license)

## Getting Started

The directory is divided in sub-folders;

-   [code](https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/tree/main/code)
    -   [Data Cleaning](https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/blob/main/code/01_Data_Cleaning.ipynb)
    -   [Data Cube Integration & Slicing](https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/blob/main/code/02_DataCube_Integration%26Slicing.ipynb)
    -   [Mining Cuboid A](https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/blob/main/code/03_Mining_CuboidA.ipynb)
    -   [Mining Cuboid B](https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/blob/main/code/03_Mining_CuboidB.ipynb)
    -   [Mining Cuboid C](https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/blob/main/code/03_Mining_CuboidC.ipynb)
-   [data](https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/tree/main/data)
    -   [cleaned](https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/tree/main/data/cleaned)
    -   [cube](https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/tree/main/data/cube)
-   [figures](https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/tree/main/figures)
    -   [building_bench](https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/tree/main/figures/building_bench)
    -   [cross_blgattrib_slice](https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/tree/main/figures/cross_blgattrib_slice)
    -   [insite_view](https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/tree/main/figures/insite_view)


Just clone this repository on your computer or Fork it on GitHub. After installing dependencies from `requirements.txt` files, the code should run properly.

### Source Data

This research uses the open [building-data-genome-project-2](https://github.com/buds-lab/building-data-genome-project-2) data set, available on github. Only the raw data from the BDG2 set are extracted. 
The sets are cleaned following the steps described in the [Data Cleaning](https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/blob/main/code/01_Data_Cleaning.ipynb) notebook and stored under `data/cleaned/`.


### Data Analysis

The source code used for the analysis can be found under the [code](https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/tree/main/code) folder, in form of Jupyter notebooks.

This work follows the [Automated daily pattern filtering of measured building performance data](https://github.com/buds-lab/day-filter) method proposed by Miller et. al, using time series Symbolic Aggregate approXimation (SAX). Daily profiles are segmented, piece-wise approximated and transformed to alphabet sequences representing daily motifs and discords.

The analysis then performs pattern identification as an OnLine Analytical Mining (OLAM) process over each cuboid of the 2D lattice of the three dimensional cube. Each cuboids appeals to a specific insight windown, namely, building benchmarking (Cuboid A), in-site view (Cuboid B) and temporal drill-in analysis (Cuboid C).

<p align="center">
  <img src="https://github.com/JulienLeprince/multidimensional-building-data-cube-pattern-identification/blob/main/figures/3DCube.jpg" alt="Building Cube"/>
</p>

## Authors

**[Julien Leprince](https://github.com/JulienLeprince)**

**Prof. [Clayton Miller](https://github.com/cmiller8)**

**Prof. [Wim Zeiler](https://www.tue.nl/en/research/researchers/wim-zeiler/)**


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


