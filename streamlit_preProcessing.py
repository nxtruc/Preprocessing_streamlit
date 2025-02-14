
import pandas as pd
import mlflow
import os
import streamlit as st

# Thiết lập thư mục lưu trữ cho MLflow
mlflow.set_tracking_uri("file:./mlruns")
os.makedirs("mlruns", exist_ok=True)

# Tạo hoặc đặt experiment
experiment_name = "Titanic_Preprocessing"
mlflow.set_experiment(experiment_name)

# Đọc dữ liệu
data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(data_url)

# Xử lý dữ liệu
# Loại bỏ cột không cần thiết
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Điền giá trị thiếu
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Mã hóa biến phân loại
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Định nghĩa đường dẫn tệp dữ liệu trước khi log với MLflow
raw_data_path = "raw_titanic.csv"
processed_data_path = "processed_titanic.csv"
df.to_csv(raw_data_path, index=False)
df.to_csv(processed_data_path, index=False)

# Logging với MLflow
with mlflow.start_run():
    # Log thông tin dữ liệu
    mlflow.log_param("missing_age_filled", df['Age'].median())
    mlflow.log_param("missing_embarked_filled", df['Embarked'].mode()[0])
    mlflow.log_param("columns_after_processing", list(df.columns))
    mlflow.log_param("dataset_shape", df.shape)
    
    # Log thêm metrics
    mlflow.log_metric("age_mean", df['Age'].mean())
    mlflow.log_metric("age_std", df['Age'].std())
    mlflow.log_metric("num_missing_values", df.isnull().sum().sum())
    
    # Log tệp dữ liệu
    mlflow.log_artifact(raw_data_path)
    mlflow.log_artifact(processed_data_path)
    
    # Thêm tag mô tả
    mlflow.set_tag("version", "1.0")
    mlflow.set_tag("author", "Tên của bạn")
    
    print("Ghi nhận quá trình tiền xử lý dữ liệu thành công với các thông số, chỉ số và tệp dữ liệu.")

# Minh họa bằng Streamlit
if __name__ == "__main__":
    st.title("Tiền xử lý dữ liệu Titanic với MLflow")
    st.write("### Dữ liệu sau khi tiền xử lý")
    st.dataframe(df.head())
    st.write(f"### Kích thước dữ liệu: {df.shape}")
    st.write(f"### Số lượng giá trị bị thiếu: {df.isnull().sum().sum()}")
    
    st.write("### Thống kê tổng quan")
    st.write(df.describe())
    
    st.write("### Các cột đã được mã hóa")
    st.write(df[['Sex', 'Embarked']].head())
    
    st.write("### Tệp dữ liệu")
    st.write("- **Dữ liệu gốc:**", raw_data_path)
    st.write("- **Dữ liệu đã xử lý:**", processed_data_path)
