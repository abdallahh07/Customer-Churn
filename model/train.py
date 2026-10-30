from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def split_train_test(x, y, test_size=0.2):
    return train_test_split(x, y, test_size=test_size, stratify=y, random_state=42)


def train_model(x_train, y_train, model):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("Model", model),
    ])
    pipeline.fit(x_train, y_train)
    return pipeline