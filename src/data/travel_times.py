import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import re
import os
import pickle
from tqdm import tqdm
import requests
import time
from sklearn.cluster import KMeans

from h3 import h3
from shapely.geometry import Point, Polygon
from pyproj import Transformer

import logging
logging.basicConfig(level=logging.INFO)

from download_pois_networks import download_POIs

# ----------------------------------------- GLOBAL VARS --------------------------------------------------------

EQUAL_AREA_PROJ = '+proj=cea'
LON_LAT_PROJ = 'EPSG:4326'
MERCATOR_PROJ = 'epsg:3395'
API_KEY = "5b3ce3597851110001cf62482b4bfe101f794dca8be9b961ebbeb9b1"
REQ_FLATTENED_SHAPE_LIMIT=3500
REQ_RATE_LIMIT=40

# -------------------------------------------- DATA PATH -------------------------------------------------------

def get_raw_data_path():
    """Function to get path for raw data. Allows to retrieve correct path when calling from another script.

    Returns:
        str: path for raw data
    """
    main_path = os.getcwd().split("15_min/")[0]+"15_min"
    raw_data_path = os.path.join(main_path, "data/raw/")    
    return raw_data_path

# ------------------------------------------ HEXAGONAL GRID ----------------------------------------------------

def create_hexagonal_grid(place_name, boundary, resolution=8, crs=LON_LAT_PROJ):
    """Create hexagonal grid inside boundary.

    Args:
        boundary (GeoDataFrame): boundary polygon
        resolution (int, optional): resolution of tessellation. Defaults to 8.

    Returns:
        GeoDataFrame: geodf of hexagons
    """
    # --------------- Check file existence ------------------------

    # Get raw data path
    raw_data_path = get_raw_data_path()
    re_place_name = re.sub(r'[\s\W]+', '', place_name)
    
    # Define file path for the hexagonal grid
    file_path_hex_grid = f"{raw_data_path}{re_place_name}_r{resolution}_hex_grid.pkl"
    
    # Check if the hexagonal grid file exists
    if os.path.exists(file_path_hex_grid):
        logging.info(f"Loading existing hexagonal grid for {place_name} at resolution {resolution}")
        with open(file_path_hex_grid, 'rb') as f:
            gdf_hex = pickle.load(f)
        return gdf_hex

    # ---------------- Compute grid ------------------------------

    logging.info(f"Creating new hexagonal grid for {place_name} at resolution {resolution}")

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

    # Save the hexagonal grid
    with open(file_path_hex_grid, 'wb') as f:
        pickle.dump(gdf_hex, f)
    logging.info(f"Saved hexagonal grid for {place_name} at resolution {resolution}")

    return gdf_hex

# -------------------------------------------- CLUSTERING ------------------------------------------------------

def KMeans_clustering(centroids,n_clusters=10):
    """Perform KMeans clustering to obtain batches of close-by centroids.

    Args:
        centroids (Series): centroids of the hexagonal grid.
        batch_size (int, optional): number of clusters. Defaults to 10.

    Returns:
        list<Series>, list<list>: Lists of centroids in batches and their center.
    """
    # Extract x and y coordinates
    coords = centroids.apply(lambda c: [c.x, c.y])
    coords = coords.tolist()

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

def plot_hex_clusters(gdf_hex):
    """Plot clustering of hexagons

    Args:
        gdf_hex (GeoDataFrame): gdf of hexagons
    """
    # Calculate centroids for all hexagons using equal_area_crs
    gdf_centroids = gdf_hex.copy()
    gdf_centroids.to_crs(EQUAL_AREA_PROJ)
    centroids = gdf_centroids.geometry.centroid
    centroids = centroids.to_crs(LON_LAT_PROJ)

    # Perform clustering
    gdf_clusters = gpd.GeoDataFrame(columns=['geometry','idx'])
    centroids_batches, batch_means = KMeans_clustering(centroids,n_clusters=10)
    for idx,centroids_batch in enumerate(centroids_batches):
        gdf_temp = gpd.GeoDataFrame({'geometry':centroids_batch.values,'idx':[idx]*len(centroids_batch)})
        gdf_clusters = pd.concat([gdf_clusters, gdf_temp], ignore_index=True)
    gdf_clusters.to_crs(LON_LAT_PROJ)

    # Plot
    _,ax = plt.subplots()
    gdf_hex.plot(ax=ax)
    gdf_clusters.plot(column='idx',cmap='tab20c',ax=ax)
    plt.savefig('../../reports/figures/hexagons_clustering.pdf')

    pass

# ------------------------------------------- EXISTING DATA ----------------------------------------------------

def check_existing_data(place_name, tags, resolution):
    """Check if travel time data already exists for a place and determine which tags are missing.
    
    Args:
        place_name (str): Name of the place
        tags (dict): Dictionary of tag types and values to analyze
    
    Returns:
        tuple: (bool indicating if any files exist, dict of missing tags, existing geodf's or empty ones)
    """
    # Get raw data path
    raw_data_path = get_raw_data_path()
    re_place_name = re.sub(r'[\s\W]+', '', place_name)
    
    # Define file paths for travel times and nearest locations
    file_path_travel_time = f"{raw_data_path}{re_place_name}__r{resolution}_travel_times.pkl"
    file_path_nearest_loc = f"{raw_data_path}{re_place_name}__r{resolution}_nearest_loc.pkl"
    
    # Check if both files exist
    files_exist = os.path.exists(file_path_travel_time) and os.path.exists(file_path_nearest_loc)
    
    # If files don't exist, all tags are missing
    if not files_exist:
        return False, tags, gpd.GeoDataFrame(), gpd.GeoDataFrame()
    
    # Load existing travel times data
    try:
        with open(file_path_travel_time, 'rb') as f:
            existing_travel_data = pickle.load(f)
        with open(file_path_nearest_loc, 'rb') as f:
            existing_loc_data = pickle.load(f)
            
        # Get column names related to travel times
        time_columns = [col for col in existing_travel_data.columns if col.startswith('timeto_')]
        
        # Extract existing tags from column names
        existing_tags = {}
        for col in time_columns:
            # Parse tag from column name (format: timeto_tagkey_tagvalue)
            parts = col.replace('timeto_', '').split('_')
            if len(parts) >= 2:
                tag_key = parts[0]
                tag_value = '_'.join(parts[1:])  # Handle tag values that might contain underscores
                
                if tag_key not in existing_tags:
                    existing_tags[tag_key] = []
                
                existing_tags[tag_key].append(tag_value)
        
        # Determine missing tags
        missing_tags = {}
        for tag_key, tag_values in tags.items():
            missing_values = []
            for tag_value in tag_values:
                # Check if this tag_value exists for this tag_key
                if tag_key not in existing_tags or tag_value not in existing_tags[tag_key]:
                    missing_values.append(tag_value)
            
            if missing_values:
                missing_tags[tag_key] = missing_values
        
        return True, missing_tags, existing_travel_data, existing_loc_data
        
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        logging.warning(f"Error loading existing data: {e}")
        return False, tags
    
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
    if len(destinations)<=n_dest:
        return destinations

    # Compute Euclidean distances
    distances = np.linalg.norm(np.array(destinations) - np.array(batch_mean), axis=1)
    
    # Return closest destinations
    closest_indices = np.argsort(distances)[:n_dest]
    return [destinations[i] for i in closest_indices]

def closest_centroid_idx(centroids_batch,source_coords):
    """Find index of the closest centroid to given source.

    Args:
        centroids_batch (_type_): _description_
        source (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Get centroids coords
    centroids_coords = centroids_batch.apply(lambda c: [c.x,c.y])
    centroids_coords = centroids_coords.tolist()
    # Compute distances, argmin, associated index
    distances = np.linalg.norm(np.array(centroids_coords)-np.array(source_coords),axis=1)
    argmin = np.argmin(distances)
    idx = centroids_batch.index[argmin]
    return idx

# --------------------------------------------------------------------------------------------------

def download_nearest_pois_travel_times(gdf_hex, pois, resolution,
                                         tags={'shop': ['supermarket'], 'leisure': ['park']}, 
                                         transport_mode='foot-walking', 
                                         place_name='Vienna, Austria',
                                         keep_closest=250,
                                         ors_api_key=API_KEY,
                                         crs=LON_LAT_PROJ,
                                         areas_crs=EQUAL_AREA_PROJ
                                         ):
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
    
    # Calculate centroids for all hexagons using areas_crs
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

    # Boolean error variable
    error=False
    
    # ITERATE over filtered pois by tag
    for tag_name, filtered_pois in filtered_pois_by_tag.items():
        logging.info(f"SEARCHING FOR CLOSEST {tag_name}\n")

        # Create copies of hexagonal grid to store results
        gdf_travel_time = gdf_hex.copy()
        gdf_travel_time = gdf_travel_time.to_crs(crs)
        gdf_nearest_loc = gdf_hex.copy()
        gdf_nearest_loc = gdf_nearest_loc.to_crs(crs)

        # If no POIs match this tag, set travel time to NaN
        if filtered_pois.empty:
            gdf_travel_time[f"timeto_{tag_name}"] = np.nan
            gdf_nearest_loc[f"nearest_{tag_name}"] = np.nan
            gdf_nearest_loc[f"nearest_{tag_name}"] = gdf_nearest_loc[f"nearest_{tag_name}"].astype('object')
            logging.warning(f"No POIs found for tag {tag_name}")
            continue
            
        # Create columns to store travel times and nearest POIs
        gdf_travel_time[f"timeto_{tag_name}"] = np.nan
        gdf_nearest_loc[f"nearest_{tag_name}"] = np.nan
        gdf_nearest_loc[f"nearest_{tag_name}"] = gdf_nearest_loc[f"nearest_{tag_name}"].astype('object')
        
        # Max 3500 routes per request are allowed, 500 requests per day
        # Obtain batches of hexagons by spatial clustering
        batch_size = int((REQ_FLATTENED_SHAPE_LIMIT-1000)/keep_closest)
        n_clusters = int(len(centroids)/batch_size)
        logging.info(f"Looking for {n_clusters} clusters of size {batch_size}\n")
        centroids_batches, batches_means = KMeans_clustering(centroids,n_clusters=n_clusters)
        
        # ITERATE over hexagons in batches to respect rate limits
        for i,(centroids_batch, batch_mean) in tqdm(enumerate(zip(centroids_batches, batches_means))):

            # Store batch of centroids as origins
            coords = centroids_batch.apply(lambda c: [c.x, c.y])
            coords = coords.tolist()
            sources = coords.copy()
                
            # Store relevant POIs as destinations
            destinations = []
            for _, poi in filtered_pois.iterrows():
                if isinstance(poi.geometry, Point):
                    # For Points, use coordinates
                    destinations.append([poi.geometry.x, poi.geometry.y])
                else:
                    # For non-point geometries (ways, relations), use centroid
                    # Create a single-row GeoDataFrame from the Series
                    poi_gdf = gpd.GeoDataFrame([poi], geometry='geometry', crs=filtered_pois.crs)
                    poi_gdf = poi_gdf.to_crs(areas_crs)
                    poi_centroid = poi_gdf.geometry.iloc[0].centroid
                    # Convert back to lon-lat crs
                    poi_centroid = list(transformer.transform(poi_centroid.x, poi_centroid.y))
                    destinations.append(poi_centroid)

            # Pre-screening based on Euclidean distance
            destinations = closest_destinations_to_batch_mean(destinations,batch_mean,n_dest=keep_closest)
            #logging.info(f"\tMinimizing among {len(destinations)} destinations.\n")

            # Define full list of locations
            locations = sources+destinations

            # Prepare ORS request body
            body = {"locations": locations,
                "sources": list(range(0,len(sources))),
                "destinations": list(range(len(sources),len(locations))),
                "metrics": ['duration']}
            
            # Make API request
            try:
                call = requests.post(ors_url, json=body, headers=headers)
                results = call.json()
                logging.debug(f"API Results: {results}")
                if 'error' in results.keys() or not results['destinations'] or not results['sources'] or not results['durations']:
                    error = True
                    print(results)
                    break
                # Extract relevant data into DataFrame
                res_durations = results["durations"]
                res_sources = [tuple(s["location"]) for s in results["sources"]]
                res_destinations = [tuple(d["location"]) for d in results["destinations"]]
                logging.info(len(res_destinations))
                df_durations = pd.DataFrame(res_durations,index=res_sources,columns=res_destinations)
                
                # For each centroid, find the shortest time and location for any POI category
                for source_, duration_series in df_durations.iterrows():
                    if not duration_series.empty: 
                        # Find min and argmin
                        min_duration = duration_series.min()
                        min_location = duration_series.idxmin()
                        # Find the corresponding index in centroids_batch
                        idx = closest_centroid_idx(centroids_batch,list(source_))  # Returned locs are not exactly the same, find nearest
                        # Store results
                        gdf_travel_time.at[idx, f"timeto_{tag_name}"] = min_duration / 60  # Convert to minutes
                        gdf_nearest_loc.at[idx, f"nearest_{tag_name}"] = min_location
                
                # Time sleep to respect rate limit
                time.sleep(60/REQ_RATE_LIMIT)
                
            except requests.exceptions.RequestException as e:
                error=True
                print(f"Error with ORS API request for batch starting at index {i}: {e}")
                # Wait longer in case of API issues
                time.sleep(5)

            # Project to original crs
            gdf_travel_time = gdf_travel_time.to_crs(original_crs)
            gdf_nearest_loc = gdf_nearest_loc.to_crs(original_crs)

        if error==False:
            # tag_name has been completed
            logging.info(f"COMPUTED TRAV. TIMES AND NEAREST LOCS FOR {tag_name}\n")
            gdf_travel_time, gdf_nearest_loc = save_results(gdf_travel_time, gdf_nearest_loc, place_name, resolution)

    return gdf_travel_time, gdf_nearest_loc

# --------------------------------------------------------------------------------------------------

def save_results(gdf_travel_time, gdf_nearest_loc, place_name, resolution):
    """Save results.

    Args:
        gdf_travel_time (_type_): _description_
        gdf_nearest_loc (_type_): _description_
        place_name (_type_): _description_
    """
    # Get raw data path
    raw_data_path = get_raw_data_path()
    re_place_name = re.sub(r'[\s\W]+', '', place_name)
    
    # Define file paths for travel times and nearest locations
    file_path_travel_time = f"{raw_data_path}{re_place_name}_r{resolution}_travel_times.pkl"
    file_path_nearest_loc = f"{raw_data_path}{re_place_name}_r{resolution}_nearest_loc.pkl"
    
    # Check if both files exist
    files_exist = os.path.exists(file_path_travel_time) and os.path.exists(file_path_nearest_loc)

    # Merge if files exist
    if files_exist:
        # Load travel times
        with open(file_path_travel_time, 'rb') as f:
            existing_travel_data = pickle.load(f)
        # Dump nearest locations
        with open(file_path_nearest_loc, 'rb') as f:
            existing_loc_data = pickle.load(f)
        # Merge travel times
        gdf_travel_time = existing_travel_data.merge(gdf_travel_time,on='geometry')
        # Merge nearest locations
        gdf_nearest_loc = existing_loc_data.merge(gdf_nearest_loc,on='geometry')

    # Dump travel times
    with open(file_path_travel_time, 'wb') as f:
        pickle.dump(gdf_travel_time, f)

    # Dump nearest locations
    with open(file_path_nearest_loc, 'wb') as f:
        pickle.dump(gdf_nearest_loc, f)

    logging.info('Saved results\n')
    
    return gdf_travel_time, gdf_nearest_loc

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

def main():

    logging.info("Starting script\n")

    # Set the two parameters
    place_name = "Vienna, Austria"
    tags={'shop':['supermarket'],'leisure':['park']}
    resolution=8

    # Download POIs
    boundary, pois = download_POIs(place_name=place_name, tags=tags)
    logging.info("Downloaded POIs\n")

    # Create hexagonal grid
    gdf_hex = create_hexagonal_grid(place_name,boundary,resolution=resolution)
    plot_hex_clusters(gdf_hex)
    logging.info("Returned hexagonal grid.\n")

    # Check whether files already exist and which tags are missing
    files_exist, missing_tags, _, _ = check_existing_data(place_name,tags,resolution=resolution)

    if not missing_tags:
        logging.info("No missing tags. Terminating script.\n")
        pass
    logging.info("Missing tags. Proceeding to download data.\n")

    # Get travel times and nearest locations for missing tags
    gdf_travel_time, gdf_nearest_loc = download_nearest_pois_travel_times(
                                            gdf_hex, pois, resolution=resolution,
                                            tags = missing_tags,
                                            transport_mode='foot-walking',
                                            place_name=place_name,
                                            keep_closest=250
                                        )
    logging.info("Downloaded all travel times and nearest locations.\n")

    logging.info("Script completed")
    
    pass

if __name__ == '__main__':
    main()



