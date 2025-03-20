import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
df = pd.read_csv('train.csv')

# Select relevant features
features = ['OverallQual', 'GrLivArea', 'BedroomAbvGr', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'LotArea']
X = df[features]
y = df['SalePrice']

# Handle missing values (if any)
X = X.fillna(X.median())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'house_price_model.pkl')
print("Model trained and saved as 'house_price_model.pkl'")