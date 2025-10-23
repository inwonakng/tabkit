
import pandas as pd
import numpy as np
import pytest
from tabkit.data.transforms import Impute, Scale, Discretize, Encode, ConvertDatetime
from tabkit.data import ColumnMetadata

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
        assert 'datetime_weekday' in transformed_df.columns
        assert transformed_df['datetime_year'].iloc[0] == 2023
        assert transformed_df['datetime_month'].iloc[0] == 1
        assert transformed_df['datetime_weekday'].iloc[0] == 6  # 2023-01-01 is Sunday (6)

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


class TestInverseTransforms:
    """Test inverse transform functionality for all transform classes."""

    def test_impute_inverse(self, sample_df):
        """Impute inverse_transform should return unchanged data."""
        imputer = Impute(method='mean')
        transformed = imputer.fit_transform(sample_df.copy())
        inverse = imputer.inverse_transform(transformed)
        pd.testing.assert_frame_equal(transformed, inverse)

    def test_scale_inverse_standard(self, sample_df, sample_metadata):
        """Scale inverse should recover original values for standard scaling."""
        df = sample_df.dropna(subset=['numeric'])
        scaler = Scale(method='standard')
        scaler.fit(df, metadata=sample_metadata)

        transformed = scaler.transform(df.copy())
        inverse = scaler.inverse_transform(transformed)

        # Check that we recover the original numeric column
        pd.testing.assert_series_equal(df['numeric'], inverse['numeric'], check_exact=False, atol=1e-10)

    def test_scale_inverse_minmax(self, sample_df, sample_metadata):
        """Scale inverse should recover original values for minmax scaling."""
        df = sample_df.dropna(subset=['numeric'])
        scaler = Scale(method='minmax')
        scaler.fit(df, metadata=sample_metadata)

        transformed = scaler.transform(df.copy())
        inverse = scaler.inverse_transform(transformed)

        # Check that we recover the original numeric column
        pd.testing.assert_series_equal(df['numeric'], inverse['numeric'], check_exact=False, atol=1e-10)

    def test_discretize_inverse(self, sample_df, sample_metadata):
        """Discretize inverse should map bins to midpoints."""
        discretizer = Discretize(method='uniform', n_bins=3)
        df_no_nan = sample_df.dropna(subset=['numeric'])
        discretizer.fit(df_no_nan, metadata=sample_metadata)

        transformed = discretizer.transform(df_no_nan.copy())
        inverse = discretizer.inverse_transform(transformed)

        # Should have continuous values again (midpoints of bins)
        assert inverse['numeric'].dtype == np.float64
        # Values should be within original range
        assert inverse['numeric'].min() >= df_no_nan['numeric'].min()
        assert inverse['numeric'].max() <= df_no_nan['numeric'].max()
        # Categorical columns should be unchanged
        pd.testing.assert_series_equal(df_no_nan['categorical'], inverse['categorical'])

    def test_encode_inverse_constant(self):
        """Encode inverse should recover original categorical values."""
        train_df = pd.DataFrame({'cat': ['A', 'B', 'A', 'C']})
        test_df = pd.DataFrame({'cat': ['A', 'B', 'D']})  # 'D' is unseen
        train_metadata = [ColumnMetadata.from_series(train_df['cat'])]

        encoder = Encode(method='constant', fill_val_name='unseen')
        encoder.fit(train_df, metadata=train_metadata)

        transformed = encoder.transform(test_df)
        inverse = encoder.inverse_transform(transformed)

        # Check that A, B are recovered correctly
        assert inverse['cat'].iloc[0] == 'A'
        assert inverse['cat'].iloc[1] == 'B'
        # Unseen value should map to 'unseen'
        assert inverse['cat'].iloc[2] == 'unseen'

    def test_encode_inverse_most_frequent(self):
        """Encode inverse with most_frequent should map codes back to categories."""
        train_df = pd.DataFrame({'cat': ['A', 'A', 'B', 'C']})
        train_metadata = [ColumnMetadata.from_series(train_df['cat'])]

        encoder = Encode(method='most_frequent')
        encoder.fit(train_df, metadata=train_metadata)

        transformed = encoder.transform(train_df)
        inverse = encoder.inverse_transform(transformed)

        pd.testing.assert_series_equal(train_df['cat'], inverse['cat'], check_names=False)

    def test_datetime_inverse_to_timestamp(self, sample_df, sample_metadata):
        """ConvertDatetime inverse should reconstruct datetime from timestamps."""
        df = sample_df.dropna(subset=['datetime'])
        transformer = ConvertDatetime(method='to_timestamp')
        transformer.fit(df, metadata=sample_metadata)

        transformed = transformer.transform(df.copy())
        inverse = transformer.inverse_transform(transformed)

        # Check datetime column is reconstructed
        assert pd.api.types.is_datetime64_any_dtype(inverse['datetime'])
        # Values should match (within second precision since we use integer seconds)
        pd.testing.assert_series_equal(
            df['datetime'].dt.floor('s'),
            inverse['datetime'].dt.floor('s'),
            check_names=False
        )

    def test_datetime_inverse_decompose(self, sample_df, sample_metadata):
        """ConvertDatetime inverse should reconstruct datetime from components."""
        df = sample_df.dropna(subset=['datetime'])
        transformer = ConvertDatetime(method='decompose')
        transformer.fit(df, metadata=sample_metadata)

        transformed = transformer.transform(df.copy())
        inverse = transformer.inverse_transform(transformed)

        # Check datetime column is reconstructed
        assert 'datetime' in inverse.columns
        assert pd.api.types.is_datetime64_any_dtype(inverse['datetime'])
        # Decomposed columns should be removed
        assert 'datetime_year' not in inverse.columns
        assert 'datetime_month' not in inverse.columns
        assert 'datetime_weekday' not in inverse.columns
        # Values should match (within second precision)
        pd.testing.assert_series_equal(
            df['datetime'].dt.floor('s'),
            inverse['datetime'].dt.floor('s'),
            check_names=False
        )

    def test_datetime_inverse_decompose_multiple_columns(self):
        """ConvertDatetime inverse should handle multiple datetime columns correctly."""
        # Create a dataframe with two datetime columns
        df = pd.DataFrame({
            'datetime1': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'datetime2': pd.to_datetime(['2024-06-15', '2024-06-16', '2024-06-17']),
            'numeric': [1.0, 2.0, 3.0]
        })

        # Create metadata
        metadata = [
            ColumnMetadata(name='datetime1', kind='datetime', dtype='datetime64[ns]'),
            ColumnMetadata(name='datetime2', kind='datetime', dtype='datetime64[ns]'),
            ColumnMetadata(name='numeric', kind='continuous', dtype='float64')
        ]

        transformer = ConvertDatetime(method='decompose')
        transformer.fit(df, metadata=metadata)

        transformed = transformer.transform(df.copy())

        # Verify both datetime columns were decomposed
        assert 'datetime1_year' in transformed.columns
        assert 'datetime2_year' in transformed.columns
        assert 'datetime1_weekday' in transformed.columns
        assert 'datetime2_weekday' in transformed.columns
        assert 'datetime1' not in transformed.columns
        assert 'datetime2' not in transformed.columns

        # Test inverse transform - this is where the bug occurred
        inverse = transformer.inverse_transform(transformed)

        # Check both datetime columns are reconstructed
        assert 'datetime1' in inverse.columns
        assert 'datetime2' in inverse.columns
        assert pd.api.types.is_datetime64_any_dtype(inverse['datetime1'])
        assert pd.api.types.is_datetime64_any_dtype(inverse['datetime2'])

        # All decomposed columns should be removed
        assert 'datetime1_year' not in inverse.columns
        assert 'datetime1_month' not in inverse.columns
        assert 'datetime1_weekday' not in inverse.columns
        assert 'datetime2_year' not in inverse.columns
        assert 'datetime2_month' not in inverse.columns
        assert 'datetime2_weekday' not in inverse.columns

        # Values should match (within second precision)
        pd.testing.assert_series_equal(
            df['datetime1'].dt.floor('s'),
            inverse['datetime1'].dt.floor('s'),
            check_names=False
        )
        pd.testing.assert_series_equal(
            df['datetime2'].dt.floor('s'),
            inverse['datetime2'].dt.floor('s'),
            check_names=False
        )

    def test_round_trip_pipeline(self, sample_df, sample_metadata):
        """Test that a full pipeline can be reversed."""
        # Create a simple pipeline: Scale -> Discretize
        df = sample_df.dropna()

        # Apply scaling
        scaler = Scale(method='minmax')
        scaler.fit(df, metadata=sample_metadata)
        scaled = scaler.transform(df.copy())

        # Apply discretization
        discretizer = Discretize(method='uniform', n_bins=5)
        discretizer.fit(scaled, metadata=sample_metadata)
        discretized = discretizer.transform(scaled.copy())

        # Reverse the pipeline
        inverse_discretized = discretizer.inverse_transform(discretized)
        inverse_scaled = scaler.inverse_transform(inverse_discretized)

        # Should approximately recover original (discretization loses info)
        assert inverse_scaled['numeric'].min() >= df['numeric'].min()
        assert inverse_scaled['numeric'].max() <= df['numeric'].max()
