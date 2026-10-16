from src.data import generate_demo_dataset
from src.features import make_features, FEATURES

def test_demo_dataset_and_features():
    df = generate_demo_dataset(days=10, save=False)
    assert {"timestamp", "energy_kwh"}.issubset(df.columns)
    featured = make_features(df)
    assert set(FEATURES).issubset(featured.columns)
    assert len(featured) > 0
    assert featured["energy_kwh"].ge(0).all()
