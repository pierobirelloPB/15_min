import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import os
import re
import pickle
import copy
import networkx as nx
from shapely.geometry import LineString
from h3 import h3
from shapely.geometry import Polygon
import geopandas as gpd
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import geopandas as gpd
import requests
import time
from shapely.geometry import Point
from pyproj import Transformer
import numpy as np
from tqdm import tqdm
import requests
from sklearn.cluster import KMeans
import logging

#import sys
#sys.path.append('../src/data')
from download_pois_networks import download_POIs

# Configure logging
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------------------------------

def get_raw_data_path():
    """Function to get path for raw data. Allows to retrieve correct path when calling from another script.

    Returns:
        str: path for raw data
    """
    main_path = os.getcwd().split("15_min/")[0]+"15_min"
    raw_data_path = os.path.join(main_path, "data/raw/")    
    return raw_data_path

# --------------------------------------------------------------------------------------------------

EQUAL_AREA_PROJ = '+proj=cea'
LON_LAT_PROJ = 'EPSG:4326'
MERCATOR_PROJ = 'epsg:3395'
API_KEY = "5b3ce3597851110001cf62482b4bfe101f794dca8be9b961ebbeb9b1"

# --------------------------------------------------------------------------------------------------

def create_hexagonal_grid(boundary, resolution=8, crs=LON_LAT_PROJ):
    """Create hexagonal grid inside boundary.

    Args:
        boundary (GeoDataFrame): boundary polygon
        resolution (int, optional): resolution of tessellation. Defaults to 8.

    Returns:
        GeoDataFrame: geodf of hexagons
    """
    # Store initial CRS
    original_crs = boundary.crs
    boundary_ = boundary.copy()

    # Reproject the polygon into equal area CRS
    boundary_ = boundary_.to_crs(crs) 

    # Get boundary polygon coordinates
    boundary_shape = boundary_.geometry.iloc[0]
    
    # Get hexagons that intersect with the boundary
    hex_ids = list(h3.polyfill(
        boundary_shape.__geo_interface__,
        resolution
    ))
    
    # Convert hexagons to polygons
    hex_polygons = []
    for hex_id in hex_ids:
        polygon_coords = h3.h3_to_geo_boundary(hex_id)
        polygon = Polygon(polygon_coords)
        hex_polygons.append({
            'geometry': polygon,
            'hex_id': hex_id
        })
    
    # Convert to GeoDataFrame
    gdf_hex = gpd.GeoDataFrame(hex_polygons, crs=crs)
    gdf_hex.to_crs(original_crs)

    return gdf_hex

# --------------------------------------------------------------------------------------------------

def KMeans_clustering(centroids,n_clusters=10):
    """Perform KMeans clustering to obtain batches of clos-by centroids

    Args:
        centroids (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    # Extract x and y coordinates
    coords = centroids.apply(lambda c: [c.x, c.y])
    coords = np.array(coords.tolist())

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(coords)
    labels = kmeans.labels_
    means = kmeans.cluster_centers_

    # Create a list of subseries based on clustering labels
    subseries = []
    for i in range(n_clusters):
        # Get indices where label equals the current cluster
        cluster_indices = np.where(labels == i)[0]
        # Create subseries using the original centroids Series
        subseries.append(centroids.iloc[cluster_indices])
    
    return subseries, means

# --------------------------------------------------------------------------------------------------

def closest_destinations_to_batch_mean(destinations, batch_mean, n_dest):
    """Find closest n_dest destinations to batch mean.

    Args:
        destinations (_type_): _description_
        batch_mean (_type_): _description_
        n_dest (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Compute Euclidean distances
    distances = np.linalg.norm(np.array(destinations) - np.array(batch_mean), axis=1)
    
    # Return closest destinations
    closest_indices = np.argsort(distances)[:n_dest]
    return destinations[closest_indices].tolist()

# --------------------------------------------------------------------------------------------------

def download_nearest_pois_travel_times(gdf_hex, pois,
                                         tags={'shop': ['supermarket'], 'leisure': ['park']}, 
                                         transport_mode='foot-walking', 
                                         ors_api_key=API_KEY,
                                         crs=LON_LAT_PROJ,
                                         areas_crs=EQUAL_AREA_PROJ,
                                         rate_limit=40,
                                         keep_closest=100):
    """
    Calculate travel times from each hexagon centroid to the nearest POI of each specified tag type.
    
    Args:
        gdf_hex (GeoDataFrame): Hexagonal grid covering the area of interest
        pois (GeoDataFrame): Points of Interest with multi-index (element_type, osmid)
        tags (dict): Dictionary of tag types and values to analyze (default: {'shop': ['supermarket'], 'leisure': ['park']})
        ors_api_key (str): API key for OpenRouteService
        transport_mode (str): Transport mode for routing (default: 'foot-walking')
        rate_limit (int): Maximum requests per minute to respect API rate limits (default: 40)
        
    Returns:
        GeoDataFrame: Original hexagonal grid with additional columns for travel times to nearest POIs
    """
    # ---------------- SETUP -------------------

    # Save original crs
    original_crs = gdf_hex.crs

    # Define transfomer
    transformer = Transformer.from_crs(areas_crs,crs)

    # Create copies of hexagonal grid to store results
    gdf_travel_time = gdf_hex.copy()
    gdf_travel_time = gdf_travel_time.to_crs(crs)
    gdf_nearest_loc = gdf_hex.copy()
    gdf_nearest_loc = gdf_nearest_loc.to_crs(crs)
    
    # Calculate centroids for all hexagons using equal_area_crs
    gdf_centroids = gdf_hex.copy()
    gdf_centroids.to_crs(areas_crs)
    centroids = gdf_centroids.geometry.centroid
    centroids = centroids.to_crs(crs)

    # Project also pois to crs
    pois_ = pois.copy()
    pois_ = pois_.to_crs(crs)
    
    # Setup for ORS requests
    ors_url = f"https://api.openrouteservice.org/v2/matrix/{transport_mode}"
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': ors_api_key,
        'Content-Type': 'application/json; charset=utf-8'
    }
    
    # Filter POIs by each tag type and value, and store in dictionary
    filtered_pois_by_tag = {}
    for tag_key, tag_values in tags.items():
        for tag_value in tag_values:
            tag_name = f"{tag_key}_{tag_value}"
            # Filter POIs that match this tag
            mask = pois_[tag_key] == tag_value
            if mask.any():
                filtered_pois_by_tag[tag_name] = pois_[mask]

    # ---------------- REQUEST DURATIONS -------------------
    
    # ITERATE over filtered pois by tag
    for tag_name, filtered_pois in filtered_pois_by_tag.items():
        # If no POIs match this tag, set travel time to NaN
        if filtered_pois.empty:
            gdf_travel_time[f"timeto_{tag_name}"] = np.nan
            gdf_nearest_loc[f"nearest_{tag_name}"] = np.nan
            continue
            
        # Create columns to store travel times and nearest POIs
        gdf_travel_time[f"timeto_{tag_name}"] = np.nan
        gdf_nearest_loc[f"nearest_{tag_name}"] = np.nan
        
        # Max 3500 routes per request are allowed, 500 requests per day
        # Obtain batches of hexagons by spatial clustering
        batch_size = int(3500-100)/keep_closest
        n_clusters = int(len(centroids)/batch_size)
        batches_centroids, batches_means = KMeans_clustering(centroids,n_clusters=n_clusters)
        
        # ITERATE over hexagons in batches to respect rate limits
        for i,batch_centroids in tqdm(enumerate(batches_centroids)):
            batch_mean = batches_means[i]

            # Store batch of centroids as origins
            coords = centroids.apply(lambda c: [c.x, c.y])
            coords = np.array(coords.tolist())
            sources = coords.copy()
                
            # Store relevant POIs as destinations
            destinations = []
            for _, poi in filtered_pois.iterrows():
                if isinstance(poi.geometry, Point):
                    # For Points, use coordinates
                    destinations.append([poi.geometry.x, poi.geometry.y])
                else:
                    # For non-point geometries (ways, relations), use centroid
                    poi_ = poi.copy()
                    poi_.to_crs(areas_crs)
                    poi_centroid = poi_.geometry.centroid
                    poi_centroid = list(transformer.transform(poi_centroid))
                    destinations.append(poi_centroid)

            # Pre-screening based on Euclidean distance
            destinations = closest_destinations_to_batch_mean(destinations,batch_mean,n_dest=keep_closest)

            # Define full list of locations
            locations = sources+destinations

            print(len(destinations))

            # Prepare ORS request body
            body = {"locations": locations,
                "sources": list(range(0,len(sources))),
                "destinations": list(range(len(sources),len(locations))),
                "metrics": ['duration']}
            
            # Make API request
            try:
                call = requests.post(ors_url, json=body, headers=headers)
                results = call.json()
                print(results)
                # Extract relevant data into DataFrame
                res_durations = results["durations"]
                res_sources = [tuple(s["location"]) for s in results["sources"]]
                res_destinations = [tuple(d["location"]) for d in results["destinations"]]
                df_durations = pd.DataFrame(res_durations,index=res_sources,columns=res_destinations)
                
                # For each centroid, find the shortest time to any POI
                for idx, duration_series in df_durations.iterrows():
                    if duration_series:  # If we have any valid durations
                        min_duration = duration_series.min()
                        min_location = duration_series.index(duration_series.argmin())
                        gdf_travel_time.loc[batch_centroids.index[idx], f"timeto_{tag_name}"] = min_duration / 60  # Convert to minutes
                        gdf_nearest_loc.loc[batch_centroids.index[idx], f"nearest_{tag_name}"] = min_location
                
                # Respect rate limits
                time.sleep(60 / rate_limit)
                
            except requests.exceptions.RequestException as e:
                print(f"Error with ORS API request for batch starting at index {i}: {e}")
                # Wait longer in case of API issues
                time.sleep(5)

            # Project to desired return_crs
            gdf_travel_time = gdf_travel_time.to_crs(original_crs)
            gdf_nearest_loc = gdf_nearest_loc.to_crs(original_crs)
    
    return gdf_travel_time, gdf_nearest_loc

# --------------------------------------------------------------------------------------------------

def save_results(gdf_travel_time, gdf_nearest_loc,place_name):
    raw_data_path = get_raw_data_path()
    # Dump travel times
    file_path_travel_time = f"{raw_data_path}{place_name}_travel_times.pkl"
    with open(file_path_travel_time, 'wb') as f:
        pickle.dump(gdf_travel_time, f)
    # Dump nearest locations
    file_path_nearest_loc = f"{raw_data_path}{place_name}_nearest_loc.pkl"
    with open(ffileile)

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

def main():

    place_name = "Vienna, Austria"

    # Download POIs
    tags={'shop':['supermarket'],'leisure':['park']}
    boundary, pois = download_POIs(place_name=place_name, tags=tags)

    # Create hexagonal grid
    gdf_hex = create_hexagonal_grid(boundary)

    # Get travel times and nearest locations
    gdf_travel_time, gdf_nearest_loc = download_nearest_pois_travel_times(
                                            gdf_hex, pois,
                                            tags = {'shop':['supermarket'], 'leisure':['park']},
                                            transport_mode='foot-walking'
                                        )
    
    save_results(gdf_travel_time, gdf_nearest_loc)
    
    pass

if __name__ == '__main__':
    main()



