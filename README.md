# Isochrones and services accessibility in Vienna.

How far is Vienna from being a 15-minute city? Project to collect POIs, routing and accessibility data.

Scripts - in src/data/:
* download_pois_networks.py -> download POIs and networks from Open Street Maps.
* travel_times.py -> create a grid tessellation for the selected city; per each grid cell, obtain and save location of the nearest POI of given category, and the time required to reach it. 

Notebooks - in notebooks/:
* clustering centroids.ipynb -> a comparison of two methods to group cells into clusters; this is used to reduce the number of API calls.
* download_pois_networks_example.ipynb -> read and visualize the data collected by the homonymous script.
* travel_times_example.ipynb -> read and visualize the data collected by the homonimous script.
* isochrone_function.ipynb -> obtain isochrones centered in given POIs, compute their unions and intersections, visualize on map.

Data - in data/raw/:
* ViennaAustria.pkl -> POIs.
* ViennaAustria_networks.pkl -> street networks.
* ViennaAustria_r8_hex_grif.pkl -> hexagonal grid.
* ViennaAustria_r8_nearest_loc -> nearest locations.
* ViennaAustria_r8_travel_times -> travel times.

Additional data - in notebooks/:
* isochrone_overlap_map.html -> isochrones map.
* isochrone_results.json -> results from isochrone_function.ipynb.


## Repository structure

```
.
├── AUTHORS.md
├── LICENSE
├── README.md
├── bin                <- Your compiled model code can be stored here (not tracked by git)
├── config             <- Configuration files, e.g., for doxygen or for your model if needed
├── **data**
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── **raw**            <- The original, immutable data dump.
├── docs               <- Documentation, e.g., doxygen or scientific papers (not tracked by git)
├── **notebooks**          <- Ipython or R notebooks
├── reports            <- For a manuscript source, e.g., LaTeX, Markdown, etc., or any project reports
│   └── figures        <- Figures for the manuscript or reports
└── **src**                <- Source code for this project
    ├── **data**           <- scripts and programs to process data
    ├── external       <- Any external source code, e.g., pull other git projects, or external libraries
    ├── models         <- Source code for your own model
    ├── tools          <- Any helper scripts go here
    └── visualization  <- Scripts for visualisation of your results, e.g., matplotlib, ggplot2 related.
```
