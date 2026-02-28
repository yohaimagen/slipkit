import pandas as pd
import numpy as np
from pyproj import CRS, Transformer
from typing import Union, List, Optional
from pathlib import Path

from slipkit.core.data import GeodeticDataSet

class GnssParser:
    """
    Parser for GNSS displacement data (CSV format).

    Handles reading CSV files, coordinate projection (WGS84 -> Local UTM), 
    and generating displacement vectors for East, North, and Up components.
    """

    @staticmethod
    def read_csv(
        filepath: Union[str, Path],
        origin_lon: float,
        origin_lat: float,
        name: str = "GNSS_Dataset",
        components: List[str] = ['E', 'N', 'U'],
        column_mapping: Optional[dict] = None
    ) -> GeodeticDataSet:
        """
        Reads a GNSS displacement CSV and converts it to a GeodeticDataSet.

        Default CSV columns: (Station, Lat, Lon, E, N, U, SigE, SigN, SigU)

        Args:
            filepath: Path to the GNSS CSV file.
            origin_lon: Longitude of the local coordinate system origin.
            origin_lat: Latitude of the local coordinate system origin.
            name: Identifier for the dataset.
            components: List of components to include ('E', 'N', 'U').
            column_mapping: Dictionary to map expected columns to CSV columns.
                Default: {'Station': 'Station', 'Lat': 'Lat', 'Lon': 'Lon', 
                          'E': 'E', 'N': 'N', 'U': 'U', 
                          'SigE': 'SigE', 'SigN': 'SigN', 'SigU': 'SigU'}

        Returns:
            GeodeticDataSet: Populated data container with local coordinates.
        """
        df = pd.read_csv(filepath)
        
        # Default mapping
        mapping = {
            'Station': 'Station', 'Lat': 'Lat', 'Lon': 'Lon',
            'E': 'E', 'N': 'N', 'U': 'U',
            'SigE': 'SigE', 'SigN': 'SigN', 'SigU': 'SigU'
        }
        if column_mapping:
            mapping.update(column_mapping)

        # Coordinate Conversion (WGS84 -> Local UTM)
        lons = df[mapping['Lon']].values
        lats = df[mapping['Lat']].values

        # UTM Projection setup (similar to InsarParser)
        utm_zone = int((origin_lon + 180) / 6) + 1
        hemisphere = 'south' if origin_lat < 0 else 'north'
        
        source_crs = CRS.from_epsg(4326)
        target_crs = CRS.from_proj4(
            f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"
        )
        
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

        # Transform points
        easting, northing = transformer.transform(lons, lats)
        # Transform Origin
        origin_e, origin_n = transformer.transform(origin_lon, origin_lat)

        # Local Cartesian (km)
        local_x = (easting - origin_e) * 1e-3
        local_y = (northing - origin_n) * 1e-3
        local_z = np.zeros_like(local_x) # Z=0 for now

        # Prepare for stacking
        all_coords = []
        all_data = []
        all_unit_vecs = []
        all_sigma = []

        num_stations = len(df)

        # Unit vectors for E, N, U
        unit_map = {
            'E': np.array([1.0, 0.0, 0.0]),
            'N': np.array([0.0, 1.0, 0.0]),
            'U': np.array([0.0, 0.0, 1.0]) # TODO - consider if Up should be flipped to match greens function convention
        }

        for comp in components:
            if comp not in ['E', 'N', 'U']:
                raise ValueError(f"Invalid component: {comp}")
            
            sig_key = f"Sig{comp}"
            csv_comp = mapping[comp]
            csv_sig = mapping[sig_key]
            
            # coords
            all_coords.append(np.column_stack((local_x, local_y, local_z)))
            # data
            all_data.append(df[csv_comp].values)
            # unit_vecs
            u_vec = unit_map[comp]
            all_unit_vecs.append(np.tile(u_vec, (num_stations, 1)))
            # sigma
            all_sigma.append(df[csv_sig].values)

        # Combine everything
        coords_final = np.vstack(all_coords)
        data_final = np.concatenate(all_data)
        unit_vecs_final = np.vstack(all_unit_vecs)
        sigma_final = np.concatenate(all_sigma)

        return GeodeticDataSet(
            coords=coords_final,
            data=data_final,
            unit_vecs=unit_vecs_final,
            sigma=sigma_final,
            name=name
        )
