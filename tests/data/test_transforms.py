
import pandas as pd
import numpy as np
import pytest
from tabkit.data.transforms import Impute, Scale, Discretize, Encode, ConvertDatetime
from tabkit.data.column_metadata import ColumnMetadata

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'numeric': [1.0, 2.0, np.nan, 4.0, 5.0],
        'categorical': ['A', 'B', 'A', 'C', np.nan],
        'constant': [1, 1, 1, 1, 1],
        'datetime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', np.nan])
    })

@pytest.fixture
def sample_metadata(sample_df):
    meta = []
    for col in sample_df.columns:
        if pd.api.types.is_numeric_dtype(sample_df[col]):
            kind = 'continuous'
        elif pd.api.types.is_datetime64_any_dtype(sample_df[col]):
            kind = 'datetime'
        else:
            kind = 'categorical'
        meta.append(ColumnMetadata(name=col, kind=kind, dtype=str(sample_df[col].dtype)))
    return meta

class TestImpute:
    def test_impute_mean(self, sample_df):
        imputer = Impute(method='mean')
        transformed_df = imputer.fit_transform(sample_df)
        assert transformed_df['numeric'].isnull().sum() == 0
        assert transformed_df['numeric'].iloc[2] == pytest.approx((1+2+4+5)/4)
        assert transformed_df['categorical'].isnull().sum() == 1

    def test_impute_most_frequent(self, sample_df):
        imputer = Impute(method='most_frequent')
        transformed_df = imputer.fit_transform(sample_df)
        assert transformed_df['categorical'].isnull().sum() == 0
        assert transformed_df['categorical'].iloc[4] == 'A'

    def test_impute_constant(self, sample_df):
        imputer = Impute(method='constant', fill_value=-1)
        transformed_df = imputer.fit_transform(sample_df)
        assert transformed_df['numeric'].iloc[2] == -1
        assert transformed_df['categorical'].iloc[4] == -1

    def test_impute_all_nan(self):
        df = pd.DataFrame({'all_nan': [np.nan, np.nan, np.nan]})
        imputer = Impute(method='mean')
        transformed_df = imputer.fit_transform(df)
        assert transformed_df['all_nan'].isnull().all()

    def test_impute_no_nan(self, sample_df):
        df = sample_df.dropna()
        imputer = Impute(method='mean')
        transformed_df = imputer.fit_transform(df.copy())
        pd.testing.assert_frame_equal(df, transformed_df)

class TestScale:
    def test_scale_standard(self, sample_df, sample_metadata):
        df = sample_df.dropna(subset=['numeric'])
        scaler = Scale(method='standard')
        transformed_df = scaler.fit(df, metadata=sample_metadata).transform(df.copy())
        assert transformed_df['numeric'].mean() == pytest.approx(0.0)
        assert transformed_df['numeric'].std(ddof=0) == pytest.approx(1.0)

    def test_scale_minmax(self, sample_df, sample_metadata):
        df = sample_df.dropna(subset=['numeric'])
        scaler = Scale(method='minmax')
        transformed_df = scaler.fit(df, metadata=sample_metadata).transform(df.copy())
        assert transformed_df['numeric'].min() == pytest.approx(0.0)
        assert transformed_df['numeric'].max() == pytest.approx(1.0)

    def test_scale_ignores_categorical(self, sample_df, sample_metadata):
        scaler = Scale(method='standard')
        transformed_df = scaler.fit_transform(sample_df.copy(), metadata=sample_metadata)
        assert 'categorical' not in scaler.scalers_
        pd.testing.assert_series_equal(sample_df['categorical'], transformed_df['categorical'], check_names=False)

    def test_scale_data_leakage(self):
        scaler = Scale(method='minmax')
        train_df = pd.DataFrame({'numeric': [10.0, 20.0], 'other': [1, 2]})
        metadata = [ColumnMetadata(name='numeric', kind='continuous', dtype='float64'), ColumnMetadata(name='other', kind='categorical', dtype='int64')]
        
        scaler.fit(train_df, metadata=metadata)
        
        test_df = pd.DataFrame({'numeric': [10.0, 30.0], 'other': [1, 2]})
        transformed_df = scaler.transform(test_df)
        
        assert transformed_df['numeric'].tolist() == [0.0, 2.0]
        assert transformed_df['other'].tolist() == [1, 2] # Should be untouched

class TestDiscretize:
    def test_discretize_uniform(self, sample_df, sample_metadata):
        discretizer = Discretize(method='uniform', n_bins=3)
        df_no_nan = sample_df.dropna(subset=['numeric'])
        transformed_df = discretizer.fit_transform(df_no_nan.copy(), metadata=sample_metadata)
        assert transformed_df['numeric'].nunique() <= 3
        assert transformed_df['numeric'].iloc[0] == 0

    def test_discretize_metadata_update(self, sample_df, sample_metadata):
        discretizer = Discretize(method='uniform', n_bins=3)
        discretizer.fit(sample_df, metadata=sample_metadata)
        new_metadata = discretizer.update_metadata(sample_df, sample_metadata)
        assert new_metadata[0].kind == 'categorical'
        assert len(new_metadata[0].mapping) == 3

class TestEncode:
    def test_encode_unseen_constant(self):
        train_df = pd.DataFrame({'cat': ['A', 'B']})
        test_df = pd.DataFrame({'cat': ['A', 'C']})
        train_metadata = [ColumnMetadata.from_series(train_df['cat'])]
        encoder = Encode(method='constant', fill_val_name='unseen')
        encoder.fit(train_df, metadata=train_metadata)
        transformed_test = encoder.transform(test_df)
        assert transformed_test['cat'].tolist() == [0, 2]

    def test_encode_data_leakage(self):
        train_df = pd.DataFrame({'cat': ['A', 'B']})
        test_df = pd.DataFrame({'cat': ['C']})
        train_metadata = [ColumnMetadata.from_series(train_df['cat'])]
        encoder = Encode(method='most_frequent')
        encoder.fit(train_df, metadata=train_metadata)
        mode_encoding = encoder.encodings_['cat'][1]
        transformed_test = encoder.transform(test_df)
        assert transformed_test['cat'].iloc[0] == mode_encoding

    def test_encode_metadata_update(self):
        train_df = pd.DataFrame({'cat': ['A', 'B']})
        train_metadata = [ColumnMetadata.from_series(train_df['cat'])]
        encoder = Encode(method='constant', fill_val_name='unseen')
        encoder.fit(train_df, metadata=train_metadata)
        new_metadata = encoder.update_metadata(train_df, train_metadata)
        assert new_metadata[0].kind == 'binary'
        assert new_metadata[0].mapping == ['A', 'B', 'unseen']

class TestConvertDatetime:
    def test_convert_datetime_to_timestamp(self, sample_df, sample_metadata):
        transformer = ConvertDatetime(method='to_timestamp')
        transformed_df = transformer.fit_transform(sample_df.copy(), metadata=sample_metadata)
        assert transformed_df['datetime'].dtype == 'int64'
        assert transformed_df['datetime'].iloc[0] == 1672531200

    def test_convert_datetime_decompose(self, sample_df, sample_metadata):
        transformer = ConvertDatetime(method='decompose')
        transformed_df = transformer.fit_transform(sample_df.copy(), metadata=sample_metadata)
        assert 'datetime' not in transformed_df.columns
        assert 'datetime_year' in transformed_df.columns
        assert transformed_df['datetime_year'].iloc[0] == 2023
        assert transformed_df['datetime_month'].iloc[0] == 1

    def test_convert_datetime_ignore(self, sample_df, sample_metadata):
        transformer = ConvertDatetime(method='ignore')
        transformed_df = transformer.fit_transform(sample_df.copy(), metadata=sample_metadata)
        assert 'datetime' not in transformed_df.columns

    def test_convert_datetime_errors_coerce(self, sample_metadata):
        df = pd.DataFrame({'datetime': ['2023-01-01', 'not-a-date']})
        # Manually create correct metadata for this test case
        dt_metadata = [ColumnMetadata(name='datetime', kind='datetime', dtype='object')]
        transformer = ConvertDatetime(method='to_timestamp')
        transformed_df = transformer.fit_transform(df, metadata=dt_metadata)
        # pd.to_numeric on a NaT gives a large negative number
        assert transformed_df['datetime'].iloc[1] < 0
