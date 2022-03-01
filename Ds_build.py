try:
    import gdal
except:
    from osgeo import gdal

class _create_ds():

    def __init__(self, ds):
        self.Projection = ds.GetProjection()
        self.GeoTransform = ds.GetGeoTransform()
        self.RasterCount = ds.RasterCount
        self.RasterXSize = ds.RasterXSize
        self.RasterYSize = ds.RasterYSize
        self.DataType = ds.GetRasterBand(1).DataType
        self.update_bbox()

    def GetProjection(self):
        return(self.Projection)

    def GetGeoTransform(self):
        return(self.GeoTransform)

    def RasterCount(self):
        return(self.RasterCount)

    def RasterXSize(self):
        return(self.RasterXSize)

    def RasterYSize(self):
        return(self.RasterYSize)

    def DataType(self):
        return(self.DataType)

    def update_bbox(self):
        self.ul_lon, self.cell_xsize, _, self.ul_lat, _, self.cell_ysize = self.GeoTransform
        self.lr_lon = self.ul_lon + (self.RasterXSize * self.cell_xsize)
        self.lr_lat = self.ul_lat + (self.RasterYSize * self.cell_ysize)

        self.bbox = tuple([[self.ul_lon, self.lr_lat], [self.lr_lon, self.ul_lat]])



def main(file):
    ds = gdal.Open(file)
    ds = _create_ds(ds)
    return ds