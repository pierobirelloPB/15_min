import osmnx as ox
import geopandas as gpd
import pandas as pd
from h3 import h3
import folium
import requests
from shapely.geometry import Polygon, Point
import numpy as np
from branca.colormap import LinearColormap
import json
from tqdm import tqdm

# Set up the OSRM API endpoint
OSRM_ENDPOINT = "http://router.project-osrm.org/table/v1/driving/"

def create_hexagonal_grid(boundary, resolution=8):
    """Create hexagonal grid over Vienna"""
    # Get boundary polygon coordinates
    boundary_shape = boundary.geometry.iloc[0]
    
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
    
    return gpd.GeoDataFrame(hex_polygons)

def calculate_access_times(hexagons, pois, service_type):
    """Calculate access times from hexagon centroids to nearest service"""
    results = []
    
    # Filter POIs by service type
    if service_type in ['supermarket']:
        service_pois = pois[pois['shop'] == service_type]
    elif service_type in ['park']:
        service_pois = pois[pois['leisure'] == service_type]
    else:
        service_pois = pois[pois['amenity'] == service_type]
    
    for idx, hex_row in tqdm(hexagons.iterrows(), total=len(hexagons)):
        centroid = hex_row.geometry.centroid
        
        # Build coordinates string starting with hexagon centroid
        coordinates = f"{centroid.y},{centroid.x}"
        
        # Add coordinates of each POI (using centroids for polygons)
        for _, poi in service_pois.iterrows():
            if poi.geometry.geom_type == 'Point':
                poi_y, poi_x = poi.geometry.y, poi.geometry.x
            else:
                # For Polygons or other geometry types, use the centroid
                poi_y, poi_x = poi.geometry.centroid.y, poi.geometry.centroid.x
            coordinates += f";{poi_y},{poi_x}"
        
        # Call OSRM API
        url = f"{OSRM_ENDPOINT}{coordinates}?annotations=duration"
        response = requests.get(url).json()
        
        # Get minimum access time
        if 'durations' in response:
            min_time = min(response['durations'][0][1:])  # Skip first value (distance to self)
            results.append({
                'hex_id': hex_row['hex_id'],
                'access_time': min_time / 60  # Convert to minutes
            })
    
    return pd.DataFrame(results)

def create_accessibility_map(hexagons, access_times, service_type):
    """Create choropleth map of accessibility"""
    # Create base map
    m = folium.Map(
        location=[48.2082, 16.3738],  # Vienna coordinates
        zoom_start=11
    )
    
    # Create colormap
    vmin = access_times['access_time'].min()
    vmax = access_times['access_time'].max()
    colormap = LinearColormap(
        colors=['green', 'yellow', 'red'],
        vmin=vmin,
        vmax=vmax
    )
    
    # Add hexagons to map
    for idx, hex_row in hexagons.iterrows():
        time = access_times[access_times['hex_id'] == hex_row['hex_id']]['access_time'].iloc[0]
        folium.GeoJson(
            hex_row.geometry.__geo_interface__,
            style_function=lambda x, time=time: {
                'fillColor': colormap(time),
                'fillOpacity': 0.7,
                'color': 'none'
            }
        ).add_to(m)
    
    # Add colormap to map
    colormap.add_to(m)
    
    return m

def main():
    # Download data
    print("Downloading Vienna data...")
    vienna, pois = download_vienna_data()
    
    # Create hexagonal grid
    print("Creating hexagonal grid...")
    hexagons = create_hexagonal_grid(vienna)
    
    # Define services to analyze
    services = [
        'hospital',
        'pharmacy',
        'school',
        'supermarket',
        'park'
    ]
    
    # Calculate access times and create maps for each service
    for service in services:
        print(f"Analyzing accessibility for {service}s...")
        access_times = calculate_access_times(hexagons, pois, service)
        
        # Create and save map
        print(f"Creating map for {service}s...")
        map_service = create_accessibility_map(hexagons, access_times, service)
        map_service.save(f'vienna_{service}_accessibility.html')
        
        # Save access times data
        access_times.to_csv(f'vienna_{service}_access_times.csv', index=False)

if __name__ == "__main__":
    main()