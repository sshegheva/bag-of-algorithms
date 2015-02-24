
def make_waldo_optimization_problem(waldo_df):
    return WaldoOpt(waldo_df).waldo_location_map


class WaldoOpt:
    def __init__(self, waldo_df):
        self.waldo_location_map = dict()
        self.waldo_df = waldo_df
        self.create_waldo_lookup()

    def create_waldo_lookup(self):
        for i, record in self.waldo_df.iterrows():
            key = "B%dP%d" % (record.Book, record.Page)
            self.waldo_location_map[key] = (record.X, record.Y)

