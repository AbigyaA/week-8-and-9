# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipaddress
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings("ignore")

# Load Datasets
fraud_df = pd.read_csv('Fraud_Data.csv')
ip_df = pd.read_csv('IpAddress_to_Country.csv')

# 1. Handle Missing Values
print("Missing Values:\n", fraud_df.isnull().sum())

# Drop rows or impute if needed (here assumed no missing values)
# fraud_df.dropna(inplace=True)

# 2. Data Cleaning
fraud_df.drop_duplicates(inplace=True)

# Convert datetime fields
fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])

# 3. Exploratory Data Analysis (EDA)
# --- Univariate
fraud_df['age'].hist(bins=30)
plt.title("User Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

sns.countplot(x='class', data=fraud_df)
plt.title("Class Distribution (0 = Not Fraud, 1 = Fraud)")
plt.show()

# --- Bivariate
sns.boxplot(x='class', y='purchase_value', data=fraud_df)
plt.title("Purchase Value vs Class")
plt.show()

# 4. Merge Datasets for Geolocation Analysis
# Convert IP address to integer
fraud_df['ip_int'] = fraud_df['ip_address'].apply(lambda x: int(ipaddress.IPv4Address(x)))

# Convert IP range columns to integers
ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].apply(lambda x: int(ipaddress.IPv4Address(x)))
ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].apply(lambda x: int(ipaddress.IPv4Address(x)))

# Map IP to Country
def map_ip_to_country(ip):
    match = ip_df[(ip_df['lower_bound_ip_address'] <= ip) & (ip_df['upper_bound_ip_address'] >= ip)]
    return match['country'].values[0] if not match.empty else 'Unknown'

fraud_df['country'] = fraud_df['ip_int'].apply(map_ip_to_country)

# 5. Feature Engineering

# Transaction frequency per user
fraud_df['user_transaction_count'] = fraud_df.groupby('user_id')['purchase_time'].transform('count')

# Hour of day & Day of week
fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek

# Time since signup (in hours)
fraud_df['time_since_signup'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds() / 3600

# 6. Data Transformation

# --- Class Imbalance Check
print("Class Distribution:\n", fraud_df['class'].value_counts(normalize=True))

# Prepare features and target
X = fraud_df.drop(['class', 'signup_time', 'purchase_time', 'ip_address', 'user_id', 'device_id'], axis=1)
y = fraud_df['class']

# Encode categorical features
X = pd.get_dummies(X, columns=['browser', 'source', 'sex', 'country'], drop_first=True)

# --- Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# --- Handle Imbalance (Choose one)
# SMOTE - Oversampling
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Random Undersampling
# rus = RandomUnderSampler(random_state=42)
# X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

# --- Normalize Numerical Features
scaler = StandardScaler()
num_cols = ['purchase_value', 'age', 'user_transaction_count', 'hour_of_day', 'day_of_week', 'time_since_signup']
X_train_res[num_cols] = scaler.fit_transform(X_train_res[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Final Dataset Check
print("Final shape (after sampling):", X_train_res.shape)
print("Sample features:\n", X_train_res.head())
