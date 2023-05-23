import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

data = pd.read_csv("hotel_bookings - hotel_bookings.csv")


imputer = SimpleImputer(strategy='mean')
data['children'] = imputer.fit_transform(data['children'].values.reshape(-1, 1))
data['country'] = data['country'].fillna('Unknown')
data['distribution_channel'] = data['distribution_channel'].fillna('Unknown')
data['is_repeated_guest'] = imputer.fit_transform(data['is_repeated_guest'].values.reshape(-1, 1))
data['previous_cancellations'] = imputer.fit_transform(data['previous_cancellations'].values.reshape(-1, 1))
data['previous_bookings_not_canceled'] = imputer.fit_transform(data['previous_bookings_not_canceled'].values.reshape(-1, 1))
data['reserved_room_type'] = data['reserved_room_type'].fillna('Unknown')
data['assigned_room_type'] = data['assigned_room_type'].fillna('Unknown')
data['booking_changes'] = imputer.fit_transform(data['booking_changes'].values.reshape(-1, 1))
data['deposit_type'] = data['deposit_type'].fillna('Unknown')
data['agent'] = imputer.fit_transform(data['agent'].values.reshape(-1, 1))
data['company'] = imputer.fit_transform(data['company'].values.reshape(-1, 1))
data['days_in_waiting_list'] = imputer.fit_transform(data['days_in_waiting_list'].values.reshape(-1, 1))
data['customer_type'] = data['customer_type'].fillna('Unknown')
data['adr'] = imputer.fit_transform(data['adr'].values.reshape(-1, 1))
data['required_car_parking_spaces'] = imputer.fit_transform(data['required_car_parking_spaces'].values.reshape(-1, 1))
data['total_of_special_requests'] = imputer.fit_transform(data['total_of_special_requests'].values.reshape(-1, 1))
data['reservation_status'] = data['reservation_status'].fillna('Unknown')


data_encoded = pd.get_dummies(data, columns=['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel',
                                             'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type',
                                             'reservation_status'])


data_encoded = data_encoded.drop('reservation_status_date', axis=1)

scaler = StandardScaler()
numerical_cols = ['lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',
                  'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
                  'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'booking_changes',
                  'agent', 'company', 'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
                  'total_of_special_requests']
data_encoded[numerical_cols] = scaler.fit_transform(data_encoded[numerical_cols])

# K-means Clustering
kmeans = KMeans(n_clusters=4)  
kmeans.fit(data_encoded.drop('is_canceled', axis=1))

wcss = []
for i in range(1, 11):  #testing loop different numbers of clusters from 1 to 10
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data_encoded.drop('is_canceled', axis=1))
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# Visualize the results
plt.scatter(data_encoded['lead_time'], data_encoded['adr'], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red')
plt.xlabel('Lead Time')
plt.ylabel('ADR')
plt.title('K-means Clustering')
plt.show()

#KNN Classification
X = data_encoded.drop('is_canceled', axis=1)
y = data_encoded['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)  
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", accuracy_knn)

#SVM Classification
svm = SVC()
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')
print("SVM F1 Score:", f1_svm)

# Model Comparison and Selection
if accuracy_knn > f1_svm:
    print("KNN performs better.")
else:
    print("SVM performs better.")