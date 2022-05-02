

from Wine_predictor.dataset import WineData


class TestWineData:
    def test___init__(self):
        self.path = Path(__file__).parent.parent / "data/winequality-red.csv"
        self.df                 = None

        # input factors
        self.fixed_acidity      = None
        self.volatile_acidity   = None
        self.citric_acid        = None
        self.residual_sugar     = None
        self.chlorides          = None
        self.free_SO2           = None
        self.total_SO2          = None
        self.density            = None
        self.sulphates          = None
        self.alcohol            = None

        # output factor
        self.quality            = None

    def test_import_data(self):

        WineData.import_data()
        # Function to import csv file
        # and run initial file changes to create good data
        self.df = pd.read_csv(self.path)
        #print("read")
        self.df = self.df.rename(columns={'free sulfur dioxide': 'free_SO2'})
