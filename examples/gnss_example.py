import matplotlib.pyplot as plt
from slipkit.utils.parsers import GnssParser
from slipkit.utils.visualizers import GnssVisualizer

def main():
    # 1. Path to example data
    csv_path = "gnss_data_example.csv"
    
    # 2. Define origin for local coordinate conversion
    # We'll use the mean lon/lat or a specific point nearby
    origin_lon = -123.0
    origin_lat = 40.0
    
    # 3. Define column mapping for this specific CSV format
    # The file contains: id,lon,lat,E,N,Up,Se,Sn,Su
    mapping = {
        'Station': 'id', 
        'Lat': 'lat', 
        'Lon': 'lon',
        'E': 'E', 
        'N': 'N', 
        'U': 'Up',
        'SigE': 'Se', 
        'SigN': 'Sn', 
        'SigU': 'Su'
    }
    
    # 4. Read the data
    print(f"Reading GNSS data from {csv_path}...")
    dataset = GnssParser.read_csv(
        csv_path, 
        origin_lon=origin_lon, 
        origin_lat=origin_lat,
        column_mapping=mapping,
        name="Example_GNSS_Dataset"
    )
    
    print(f"Loaded {len(dataset)//3} stations.")
    
    # 5. Visualize the data
    print("Plotting GNSS vectors and saving to gnss_example_plot.png...")
    # Using a much larger scale (e.g., 1000) because positions are in km 
    # and displacements are in meters (fractions of a meter).
    GnssVisualizer.plot_gnss_vectors(
        dataset, 
        scale=1000.0,  
        title="Example GNSS Displacement Vectors",
        return_fig_ax=True
    )
    plt.savefig("gnss_example_plot.png", dpi=300, bbox_inches='tight')
    print("Done. Plot saved as 'gnss_example_plot.png'.")

if __name__ == "__main__":
    main()
