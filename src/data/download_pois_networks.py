import pandas as pd # type: ignore
import osmnx as ox # type: ignore
import os
import re
import pickle
import copy
import networkx as nx # type: ignore
import logging

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

def download_boundary(place_name="Vienna, Austria"):
    """Download boundary polygon for selected place.

    Args:
        place_name (str, optional): Place name. Defaults to "Vienna, Austria".

    Returns:
        GeoDataFrame: geodf with one row, containing boundary polygon.
    """
    boundary = ox.geocode_to_gdf(place_name)
    return boundary

# --------------------------------------------------------------------------------------------------

def download_POIs(place_name="Vienna, Austria", tags={'shop':['supermarket'],'leisure':['park']}):
    """Download OSM data for selected place. 
    If any data for the place has already been downloaded, only missing tags are downloaded.
    If no missing tags are found, data are loaded and returned.

    Args:
        place_name (str, optional): Place name. Defaults to "Vienna, Austria".
        tags (dict, optional): Category tags. Defaults to {'shop':['supermarket'],'leisure':['park']}.

    Returns:
        GeoDataFrame, GeoDataFrame: geodf's for place boundary and POIs.
    """
    # Get data path
    raw_data_path = get_raw_data_path()

    # Get place boundary
    boundary = download_boundary(place_name)
    
    # Place filename (\s = spaces, \W = non-word chars)
    re_place_name = re.sub(r'[\s\W]+', '', place_name)
    place_filename = f"{re_place_name}.pkl"
    file_path = f"{raw_data_path}{place_filename}"
    
    # Create empty DataFrame and tags to download dict
    new_pois = None
    tags_to_download = copy.deepcopy(tags)
    
    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            existing_pois = pickle.load(f)
        
        # Check which categories already exist in the data
        for key, values in tags.items():
            for value in values:
                # If key exists in columns and value exists in that category
                if key in existing_pois.columns and existing_pois[existing_pois[key] == value].shape[0] > 0:
                    # Remove from tags to download
                    if key in tags_to_download and value in tags_to_download[key]:
                        tags_to_download[key].remove(value)
                    # If the key has no more values, remove it
                    if key in tags_to_download and not tags_to_download[key]:
                        del tags_to_download[key]
        
        # Download missing tags if any
        if tags_to_download:
            new_pois = ox.features_from_polygon(
                boundary.geometry.iloc[0],
                tags=tags_to_download
            )
            
            # Combine with existing data (avoiding duplicates)
            if not new_pois.empty:
                pois = pd.concat([existing_pois, new_pois]).drop_duplicates()
                with open(file_path, 'wb') as f:
                    pickle.dump(pois, f)
            else:
                pois = existing_pois

        else:
            # No new categories to download
            pois = existing_pois
            
    else:
        # File doesn't exist, download all data
        pois = ox.features_from_polygon(
            boundary.geometry.iloc[0],
            tags=tags
        )
        # Save geodataframe
        with open(file_path, 'wb') as f:
            pickle.dump(pois, f)
    
    # Print count for each type
    print(f"Total POIs: {len(pois)}")
    
    # If new data was downloaded
    if new_pois is not None and not new_pois.empty:
        print(f"Newly downloaded POIs: {len(new_pois)}")
    
    # Print summary
    for key, values in tags.items():
        for value in values:
            if key in pois.columns:
                count = len(pois[pois[key] == value])
                print(f"Found {count} {key}-{value}")
    
    return boundary, pois

# --------------------------------------------------------------------------------------------------

def download_street_network(place_name="Vienna, Austria", network_types=None):
    """Download OSM street network data for selected place.
    If any network type for the place has already been downloaded, only missing types are downloaded.
    If no missing types are found, data are loaded and returned.

    Args:
        place_name (str, optional): Place name. Defaults to "Vienna, Austria".
        network_types (list, optional): List of network types to download. 
                                       Defaults to ['drive', 'bike', 'walk'].
                                       Options are: 'drive', 'bike', 'walk', 'drive_service', 'all', 'all_private'.

    Returns:
        GeoDataFrame, dict: geodf for place boundary and dictionary of networks by type.
    """
    # Get raw data path
    raw_data_path = get_raw_data_path()

    # Set default network types if None
    if network_types is None:
        network_types = ['drive', 'bike', 'walk']
    
    # Get place boundary
    boundary = download_boundary(place_name)
    
    # Place filename
    re_place_name = re.sub(r'[\s\W]+', '', place_name)
    place_filename = f"{re_place_name}_networks.pkl"
    file_path = f"{raw_data_path}{place_filename}"
    
    # Networks dict to store all downloaded networks
    networks = {}
    # Types to download
    types_to_download = copy.deepcopy(network_types)
    
    # Check if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            existing_networks = pickle.load(f)
        
        # Check which network types already exist
        for network_type in network_types:
            if network_type in existing_networks:
                # Remove from types to download
                if network_type in types_to_download:
                    types_to_download.remove(network_type)
                networks[network_type] = existing_networks[network_type]
        
        # Download missing network types if any
        if types_to_download:
            for network_type in types_to_download:
                print(f"Downloading {network_type} network...")
                # Download network
                G = ox.graph_from_polygon(
                    boundary.geometry.iloc[0],
                    network_type=network_type
                )
                networks[network_type] = G
            
            # Combine with existing networks
            all_networks = {**existing_networks, **networks}
            
            # Save the updated networks
            with open(file_path, 'wb') as f:
                pickle.dump(all_networks, f)
            
            # Update networks to include all
            networks = all_networks
        else:
            # No new network types to download
            networks = existing_networks
            
    else:
        # File doesn't exist, download all networks
        for network_type in network_types:
            print(f"Downloading {network_type} network...")
            # Download network
            G = ox.graph_from_polygon(
                boundary.geometry.iloc[0],
                network_type=network_type
            )
            networks[network_type] = G
        
        # Save networks
        with open(file_path, 'wb') as f:
            pickle.dump(networks, f)
    
    # Print summary
    for network_type, G in networks.items():
        print(f"{network_type.capitalize()} network: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    return boundary, networks

# --------------------------------------------------------------------------------------------------

def networks_to_gdf(networks):
    """Convert networks dictionary to GeoDataFrame for visualization and analysis.
    
    Args:
        networks (dict): Dictionary of networks by type
        
    Returns:
        GeoDataFrame: Combined GeoDataFrame with all networks, tagged by type
    """
    gdfs = []
    
    for network_type, G in networks.items():
        # Get edge GeoDataFrame
        gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        
        # Add network type as a column
        gdf_edges['network_type'] = network_type
        
        # Add to list
        gdfs.append(gdf_edges)
    
    # Combine all GeoDataFrames
    if gdfs:
        combined_gdf = pd.concat(gdfs)
        return combined_gdf
    else:
        return None
    
# --------------------------------------------------------------------------------------------------

def analyze_street_network(networks):
    """Analyze street networks and generate statistics.
    
    Args:
        networks (dict): Dictionary of networks by type
        
    Returns:
        dict: Dictionary of statistics by network type
    """
    stats = {}
    
    for network_type, G in networks.items():
        network_stats = {
            'node_count': len(G.nodes),
            'edge_count': len(G.edges),
            'total_edge_length_km': sum([data.get('length', 0) for u, v, data in G.edges(data=True)]) / 1000,
            'avg_edge_length_m': sum([data.get('length', 0) for u, v, data in G.edges(data=True)]) / len(G.edges) if len(G.edges) > 0 else 0,
            'avg_node_degree': sum(dict(G.degree()).values()) / len(G.nodes) if len(G.nodes) > 0 else 0,
            'connected_components': nx.number_connected_components(G.to_undirected()),
            # Street density = num edges / sum length in km
            'avg_street_density': len(G.edges) / (sum([data.get('length', 0) for u, v, data in G.edges(data=True)]) / 1000) if sum([data.get('length', 0) for u, v, data in G.edges(data=True)]) > 0 else 0
        }
        
        # Add stats to dictionary
        stats[network_type] = network_stats

    print('Computed Network Statistics')

    return stats

# --------------------------------------------------------------------------------------------------

def main():

    logging.info("Starting script")

    # Download POIs
    tags={'shop':['supermarket'],'leisure':['park']}
    boundary, pois = download_POIs(place_name="Vienna, Austria", tags=tags)

    # Download networks
    boundary, networks = download_street_network("Vienna, Austria", network_types=['drive', 'bike', 'walk'])
        
    # Convert to GeoDataFrame for visualization
    streets_gdf = networks_to_gdf(networks)

    # Analyze networks
    network_stats = analyze_street_network(networks)
    print("\nNetwork Statistics:")
    for network_type, stats in network_stats.items():
        print(f"\n{network_type.capitalize()} Network:")
        for stat, value in stats.items():
            print(f"  {stat}: {value:.2f}" if isinstance(value, float) else f"  {stat}: {value}")

    logging.info("Script completed")

if __name__ == '__main__':
    main()
