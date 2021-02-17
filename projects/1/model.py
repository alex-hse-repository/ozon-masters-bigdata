from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]
categorical_features_train = ["cf"+str(i) for i in [6,8,13,14,16,17,19,25]] 
features = numeric_features+categorical_features_train
fields = ["id", "label"] + numeric_features + categorical_features

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features_train)
    ]
)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('linearregression', LinearSVC())
])