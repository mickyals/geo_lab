import numpy as np
import torch
from torch.utils.data import Dataset
from pyDOE3 import lhs
import xarray as xr


class ERA5MultiData:
    """
    Memory-efficient rewrite:
    - no meshgrid
    - no large concatenations
    - no duplicated metadata
    - lazy xarray slicing preserved
    """

    def __init__(self, data_dir, read_data_fn, variables):
        with read_data_fn(data_dir) as ds:
            self.dimensions = dict(ds.sizes)

            coord_keep = ["valid_time", "pressure_level", "latitude", "longitude"]

            # Store only xarray DataArrays (lazy until sliced)
            self.coordinates = {}
            for c in coord_keep:
                if c not in ds.coords:
                    continue

                arr = ds.coords[c].astype(np.float32)

                if c == "longitude":
                    arr = ((arr + 180) % 360) - 180

                #if c == "pressure_level":
                #    arr = arr * 100.0  # hPa → Pa

                self.coordinates[c] = arr

            # Variables: store xarray DataArray only
            self.variables = {
                v: ds.data_vars[v].astype(np.float32) for v in variables
            }

            self.attributes = ds.attrs

        self.data_dir = data_dir

    @staticmethod
    def _make_slice(idx_range):
        if not idx_range:
            return slice(None)
        return slice(idx_range[0], idx_range[1])

    def subset_data(self, t_range, p_range, la_range, lo_range):
        """
        Slices xarray directly → numpy only after slicing.
        Output remains identical to original structure.
        """

        t = self._make_slice(t_range)
        p = self._make_slice(p_range)
        la = self._make_slice(la_range)
        lo = self._make_slice(lo_range)

        # Coordinates (1D arrays)
        out_coords = {}
        for name, arr in self.coordinates.items():
            if name == "valid_time":
                sliced = arr[t]
            elif name == "pressure_level":
                sliced = arr[p]
            elif name == "latitude":
                sliced = arr[la]
            elif name == "longitude":
                sliced = arr[lo]
            else:
                sliced = arr

            data = sliced.values  # now numpy, small
            out_coords[name] = (data, data.size, data.shape, str(data.dtype))

        # Variables (4D arrays) → flatten later
        out_vars = {}
        for v, arr in self.variables.items():
            sliced = arr.isel(
                valid_time=t, pressure_level=p, latitude=la, longitude=lo
            )
            data = sliced.values  # still reasonably sized
            out_vars[v] = (data, data.size, data.shape, str(data.dtype))

        return out_coords, out_vars

    def _get_stats(self, item):
        return {
            k: [
                v[0].min(),
                v[0].max(),
                v[0].mean(),
                v[0].std(),
            ]
            for k, v in item.items()
        }

    def _load_dataset(self):
        with xr.open_dataset(self.data_dir) as ds:
            return ds

    def generate_virtual_samples(self, num_samples, coord_stats, use_lhs=True):
        names = list(coord_stats.keys())
        lo = np.array([coord_stats[n][0] for n in names], dtype=np.float32)
        hi = np.array([coord_stats[n][1] for n in names], dtype=np.float32)

        if use_lhs:
            pts = lhs(len(names), samples=num_samples)
            pts = lo + (hi - lo) * pts
        else:
            pts = np.random.uniform(lo, hi, (num_samples, len(names))).astype(np.float32)

        return {name: pts[:, i] for i, name in enumerate(names)}

    def run(
            self,
            time_idx_range,
            pressure_idx_range,
            latitude_idx_range,
            longitude_idx_range,
            indexing="ij",
            num_samples=0,
            include_virtual=False,
            use_lhs=True,
    ):
        """
        Load ERA5 (multiple variables), extract ranges, flatten meshgrid,
        and append optional virtual samples.

        RETURNS (consistent with original codebase):
        ----------------------------------------------------
        data_dict = {
            "time": 1D array of length N,
            "pressure": 1D array of length N,
            "latitude": 1D array of length N,
            "longitude": 1D array of length N,
            "<var1>": flattened var1,
            "<var2>": flattened var2,
            ...
            "classification": boolean array (True = real, False = virtual) [IF virtual requested]
        }

        count = (N_real, N_total)

        Notes:
        - Ensures the flattened spatial-temporal indexing matches ERA5MultiDataset.
        - Prevents out-of-bounds indexing seen in your traceback.
        """

        # --------------------------------------------------------
        # 1. Load dataset with xarray
        # --------------------------------------------------------
        ds = self._load_dataset()

        # --------------------------------------------------------
        # 2. Slice coordinates
        # --------------------------------------------------------
        time = ds["valid_time"].values[
            slice(*time_idx_range) if time_idx_range else slice(None)
        ]
        pressure = ds["pressure_level"].values[
            slice(*pressure_idx_range) if pressure_idx_range else slice(None)
        ]
        lat = ds["latitude"].values[
            slice(*latitude_idx_range) if latitude_idx_range else slice(None)
        ]
        lon = ds["longitude"].values[
            slice(*longitude_idx_range) if longitude_idx_range else slice(None)
        ]
        if lon.max()>180:
            lon = ((lon + 180) % 360) - 180

        # --------------------------------------------------------
        # 3. Build the 4-D meshgrid: (time, pressure, latitude, longitude)
        #    This defines the *real data point indexing*
        # --------------------------------------------------------
        T, P, Y, X = np.meshgrid(
            time, pressure, lat, lon, indexing=indexing
        )
        N_real = T.size  # number of real samples (flattened)

        # --------------------------------------------------------
        # 4. Flatten coordinates into 1-D arrays
        # --------------------------------------------------------
        data = {
            "valid_time": T.ravel(),
            "pressure_level": P.ravel(),
            "latitude": Y.ravel(),
            "longitude": X.ravel(),
        }

        # --------------------------------------------------------
        # 5. Extract ERA5 variable arrays and flatten
        #    (Assumes ds[var] has dims (time, level, lat, lon))
        # --------------------------------------------------------
        for var in self.variables:
            if var not in ds:
                raise KeyError(f"Variable '{var}' not found in ERA5 dataset.")
            data[var] = ds[var].values[
                slice(*time_idx_range) if time_idx_range else slice(None),
                slice(*pressure_idx_range) if pressure_idx_range else slice(None),
                slice(*latitude_idx_range) if latitude_idx_range else slice(None),
                slice(*longitude_idx_range) if longitude_idx_range else slice(None)
            ].ravel()

        # --------------------------------------------------------
        # 6. Add virtual samples (optional)
        # --------------------------------------------------------
        if include_virtual and num_samples > 0:

            # user-provided sampler or LHS
            virtual = self._generate_virtual_samples(
                num_samples=num_samples,
                time=time,
                pressure=pressure,
                lat=lat,
                lon=lon,
                use_lhs=use_lhs,
            )

            # Append to real samples
            for key in data.keys():
                if key in virtual:
                    data[key] = np.concatenate([data[key], virtual[key]])

            # classification mask
            classification = np.zeros(N_real + num_samples, dtype=bool)
            classification[:N_real] = True
            data["classification"] = classification

            N_total = N_real + num_samples

        else:
            # still return count with classification ignored upstream
            N_total = N_real
            data["classification"] = np.ones(N_real, dtype=bool)



        return {
            "data": data,
            "count": (N_real, N_total),
        }

    # def run(
    #     self,
    #     time_idx_range,
    #     pressure_idx_range,
    #     latitude_idx_range,
    #     longitude_idx_range,
    #     indexing,
    #     num_samples=10000,
    #     include_virtual=False,
    #     use_lhs=True,
    # ):
    #     # (1) Subset
    #     coords_sub, vars_sub = self.subset_data(
    #         time_idx_range,
    #         pressure_idx_range,
    #         latitude_idx_range,
    #         longitude_idx_range,
    #     )
    #
    #     # (2) Statistics
    #     coord_stats = self._get_stats(coords_sub)
    #     var_stats = self._get_stats(vars_sub)
    #     statistics = {**coord_stats, **var_stats}
    #
    #     # (3) Real flattened data
    #     flat_coords = {k: v[0].ravel() for k, v in coords_sub.items()}
    #     flat_vars = {k: v[0].ravel() for k, v in vars_sub.items()}
    #
    #     first_var = next(iter(vars_sub))
    #     real_count = vars_sub[first_var][1]
    #     real_classifier = np.ones(real_count, dtype=bool)
    #
    #     # (4) Virtual data
    #     if include_virtual:
    #         vcoords = self.generate_virtual_samples(
    #             num_samples=num_samples,
    #             coord_stats=coord_stats,
    #             use_lhs=use_lhs,
    #         )
    #         vcount = len(next(iter(vcoords.values())))
    #
    #         # Vars = all zeros (lazy fallback)
    #         vvars = {v: np.zeros(vcount, dtype=np.float32) for v in vars_sub}
    #         vclass = np.zeros(vcount, dtype=bool)
    #
    #         # Combine logically (no huge intermediate arrays)
    #         data = {
    #             **{k: np.concatenate([flat_coords[k], vcoords[k]]) for k in flat_coords},
    #             **{
    #                 v: np.concatenate([flat_vars[v], vvars[v]])
    #                 for v in flat_vars
    #             },
    #             "classification": np.concatenate([real_classifier, vclass]),
    #         }
    #
    #         return (
    #             {"data": data, "count": [real_count + vcount, real_count, vcount]},
    #             statistics,
    #         )
    #
    #     # (5) Real only
    #     data = {
    #         **flat_coords,
    #         **flat_vars,
    #         "classification": real_classifier,
    #     }
    #
    #     return ({"data": data, "count": [real_count]}, statistics)


# -------------------------------------------------------------------
# PyTorch Dataset
# -------------------------------------------------------------------
class ERA5MultiDataset(Dataset):
    def __init__(self, data, statistics, indices, include_virtual, variables, pi_scale):
        self.data = data
        self.statistics = statistics
        self.idx = indices
        self.include_virtual = include_virtual
        self.variables = variables
        self.pi_scale = pi_scale

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]

        # Direct lookups
        lon = self.data["longitude"][j]
        lat = self.data["latitude"][j]
        pres = self.data["pressure_level"][j]
        time = self.data["valid_time"][j]

        # Unpack statistics
        lon_min, lon_max = self.statistics["longitude"][:2]
        lat_min, lat_max = self.statistics["latitude"][:2]
        t_min, t_max = self.statistics["valid_time"][:2]
        p_min, p_max = self.statistics["pressure_level"][:2]

        # Normalized coordinates
        lon_n = self._norm(lon, lon_min, lon_max)
        lat_n = self._norm(lat, lat_min, lat_max)
        tim_n = self._timedelta_norm(t_min, t_max, time)
        pres_n = self._norm(pres, p_min, p_max)

        if self.pi_scale:
            lon_n = lon_n * torch.pi
            lat_n = lat_n * (torch.pi / 2)

        coords = {
            "longitude": torch.tensor(lon_n, dtype=torch.float32),
            "latitude": torch.tensor(lat_n, dtype=torch.float32),
            "pressure_level": torch.tensor(pres_n, dtype=torch.float32),
            "time": torch.tensor(tim_n, dtype=torch.float32),
        }

        # Variables
        vars_out = {}
        for var in self.variables:
            vmin, vmax = self.statistics[var][:2]
            raw = self.data[var][j]
            norm = self._norm(raw, vmin, vmax)
            vars_out[var] = torch.tensor(norm, dtype=torch.float32)


        cls = torch.tensor(int(self.data["classification"][j]), dtype=torch.long)

        return {"coords": coords, "variables": vars_out, "classification": cls}

    def _norm(self, x, lo, hi):
        if lo == hi:
            return 1.0
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    def _timedelta_norm(self, t0, t1, t):
        if t0 == t1:
            return 1.0
        # Convert datetime64[ns] to numeric (nanoseconds since epoch) for arithmetic
        if hasattr(t, 'astype'):  # Check if it's a numpy array
            t = t.astype('float64')
            t0 = t0.astype('float64')
            t1 = t1.astype('float64')
        return 2 * (t - t0) / (t1 - t0) -1















#
# import numpy as np
# from pyDOE3 import lhs
# import torch
# from torch.utils.data import Dataset
#
#
# # -------------------------------------------------------------------
# # Era5 data container
# # -------------------------------------------------------------------
# class ERA5MultiData:
#     def __init__(self, data_dir, read_data_fn, variables):
#
#         with read_data_fn(data_dir) as ds:
#             self.dimensions = dict(ds.sizes)
#
#             coords_to_keep = [
#                 "valid_time",
#                 "pressure_level",
#                 "latitude",
#                 "longitude"
#             ]
#
#             # Enforce canonical tuple: (data, size, shape, dtype)
#             self.coordinates = {}
#             for c in coords_to_keep:
#                 if c in ds.coords:
#                     coord_data = ds.coords[c].data.astype(np.float32)
#
#                     if c == "longitude":
#                         coord_data = (coord_data + 180) % 360 - 180
#                     self.coordinates[c] = (
#                         coord_data,
#                         coord_data.size,
#                         coord_data.shape,
#                         str(coord_data.dtype)
#                     )
#
#             self.variables = {
#                 v: (
#                     ds.data_vars[v].astype(np.float32).values,
#                     ds.data_vars[v].size,
#                     ds.data_vars[v].shape,
#                     str(ds.data_vars[v].dtype),
#                 )
#                 for v in variables
#             }
#
#             self.attributes = ds.attrs
#
#     @staticmethod
#     def _make_slice(idx_range):
#         if not idx_range:
#             return slice(None)
#         return slice(idx_range[0], idx_range[1])
#
#     def subset_data(self, t_range, p_range, la_range, lo_range):
#
#         t = self._make_slice(t_range)
#         p = self._make_slice(p_range)
#         la = self._make_slice(la_range)
#         lo = self._make_slice(lo_range)
#
#         # Slice variables
#         out_vars = {}
#         for var, (arr, _, _, _) in self.variables.items():
#             sliced = arr[t, p, la, lo]
#             out_vars[var] = (
#                 sliced,
#                 sliced.size,
#                 sliced.shape,
#                 str(sliced.dtype),
#             )
#
#         # Slice coordinates, consistent logic
#         out_coords = {}
#         for name, (arr, _, _, _) in self.coordinates.items():
#             if name == "valid_time":
#                 sliced = arr[t]
#             elif name == "pressure_level":
#                 sliced = arr[p]
#             elif name == "latitude":
#                 sliced = arr[la]
#             elif name == "longitude":
#                 sliced = arr[lo]
#             else:
#                 sliced = arr
#
#             out_coords[name] = (
#                 sliced,
#                 sliced.size,
#                 sliced.shape,
#                 str(sliced.dtype),
#             )
#
#         return out_coords, out_vars
#
#     # Returns {coord_name: [min, max, mean, std]}
#     def _get_stats(self, item):
#         return {
#             k: [
#                 v[0].min(),
#                 v[0].max(),
#                 v[0].mean(),
#                 v[0].std(),
#             ]
#             for k, v in item.items()
#         }
#
#     def generate_virtual_samples(self, num_samples, coord_stats, use_lhs=True):
#
#         names = list(coord_stats.keys())
#         lo = np.array([coord_stats[n][0] for n in names])
#         hi = np.array([coord_stats[n][1] for n in names])
#
#         if use_lhs:
#             pts = lhs(len(names), samples=num_samples)
#             pts = lo + (hi - lo) * pts
#         else:
#             pts = np.random.uniform(lo, hi, (num_samples, len(names)))
#
#         return {name: pts[:, i] for i, name in enumerate(names)}
#
#     def generate_mesh(self, coords, indexing="ij"):
#         names = list(coords.keys())
#         arrays = [coords[n][0] for n in names]
#         meshes = np.meshgrid(*arrays, indexing=indexing)
#
#         out = {}
#         for name, arr in zip(names, meshes):
#             out[name] = (
#                 arr.ravel(),
#                 arr.size,
#                 arr.shape,
#                 str(arr.dtype),
#             )
#         return out
#
#     def run(
#         self,
#         time_idx_range,
#         pressure_idx_range,
#         latitude_idx_range,
#         longitude_idx_range,
#         indexing,
#         num_samples=10000,
#         include_virtual=False,
#         use_lhs=True,
#     ):
#
#         coords_sub, vars_sub = self.subset_data(
#             time_idx_range,
#             pressure_idx_range,
#             latitude_idx_range,
#             longitude_idx_range,
#         )
#
#         coord_stats = self._get_stats(coords_sub)
#         var_stats = self._get_stats(vars_sub)
#         statistics = {**coord_stats, **var_stats}
#
#         first_var = next(iter(vars_sub))
#         real_count = vars_sub[first_var][1]
#
#         real_coords = self.generate_mesh(coords_sub, indexing=indexing)
#         real_classifier = np.ones(real_count, dtype=bool)
#
#         if include_virtual:
#
#             vcoords = self.generate_virtual_samples(
#                 num_samples=num_samples,
#                 coord_stats=coord_stats,
#                 use_lhs=use_lhs,
#             )
#             vcount = len(next(iter(vcoords.values())))
#
#             vdata = {v: np.zeros(vcount) for v in vars_sub}
#             vclass = np.zeros(vcount, dtype=bool)
#
#             # Combine coordinates
#             combined_coords = {
#                 k: np.concatenate([real_coords[k][0], vcoords[k]])
#                 for k in real_coords
#             }
#
#             # Combine variables
#             combined_vars = {
#                 v: np.concatenate([vars_sub[v][0].ravel(), vdata[v]])
#                 for v in vars_sub
#             }
#
#             combined_class = np.concatenate([real_classifier, vclass])
#
#             return (
#                 {
#                     "data": {
#                         **combined_coords,
#                         **combined_vars,
#                         "classification": combined_class,
#                     },
#                     "count": [real_count + vcount, real_count, vcount],
#                 },
#                 statistics,
#             )
#
#         real_vars = {v: vars_sub[v][0].ravel() for v in vars_sub}
#
#         return (
#             {
#                 "data": {
#                     **{k: v[0] for k, v in real_coords.items()},
#                     **real_vars,
#                     "classification": real_classifier,
#                 },
#                 "count": [real_count],
#             },
#             statistics,
#         )
#
#
# # -------------------------------------------------------------------
# # PyTorch Dataset
# # -------------------------------------------------------------------
# class ERA5MultiDataset(Dataset):
#     def __init__(self, data, statistics, indices, include_virtual, variables, pi_scale):
#         self.data = data
#         self.statistics = statistics
#         self.idx = indices
#         self.include_virtual = include_virtual
#         self.variables = variables
#         self.pi_scale = pi_scale
#
#     def __len__(self):
#         return len(self.idx)
#
#     def __getitem__(self, i):
#
#         j = self.idx[i]
#
#         lon = self.data["longitude"][j]
#         lat = self.data["latitude"][j]
#         pres = self.data["pressure_level"][j]
#         time = self.data["valid_time"][j]
#
#         # Geo normalization uses min/max from stats
#         lon_min, lon_max = self.statistics["longitude"][:2]
#         lat_min, lat_max = self.statistics["latitude"][:2]
#         t_min, t_max = self.statistics["valid_time"][:2]
#         p_min, p_max = self.statistics["pressure_level"][:2]
#
#         # Normalized coords
#         lon_n = self._norm(lon, lon_min, lon_max)
#         lat_n = self._norm(lat, lat_min, lat_max)
#         tim_n = self._timedelta_norm(t_min, t_max, time)
#         pres_n = self._norm(pres, p_min, p_max)
#
#         if self.pi_scale:
#             lon_n = lon_n * torch.pi
#             lat_n = lat_n * (torch.pi / 2)
#
#         coords = {
#             "longitude": torch.tensor(lon_n, dtype=torch.float32),
#             "latitude": torch.tensor(lat_n, dtype=torch.float32),
#             "pressure_level": torch.tensor(pres_n, dtype=torch.float32),
#             "time": torch.tensor(tim_n, dtype=torch.float32),
#         }
#
#         vars_data = {}
#         for var in self.variables:
#             vmin, vmax = self.statistics[var][:2]
#             vraw = self.data[var][j]
#             vnorm = self._norm(vraw, vmin, vmax)
#             vars_data[var] = torch.tensor(vnorm, dtype=torch.float32)
#
#         cls = torch.tensor(int(self.data["classification"][j]), dtype=torch.long)
#
#         return {"coords": coords, "variables": vars_data, "classification": cls}
#
#     # Why: consistent [-1, 1] normalization
#     def _norm(self, x, lo, hi):
#         if lo == hi:
#             return 1.0
#         return 2.0 * (x - lo) / (hi - lo) - 1.0
#
#     # Why: time normalized to [0,1]
#     def _timedelta_norm(self, t0, t1, t):
#         if t0 == t1:
#             return 1.0
#         return (t - t0) / (t1 - t0)
#
