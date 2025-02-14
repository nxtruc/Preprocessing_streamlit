import pandas as pd
import os
import streamlit as st

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

# Định nghĩa đường dẫn tệp dữ liệu trước khi log
data_path = "processed_titanic.csv"
df.to_csv(data_path, index=False)

# Minh họa bằng Streamlit
if __name__ == "__main__":
    st.title("Tiền xử lý dữ liệu Titanic")
    st.write("### Dữ liệu sau khi tiền xử lý")
    st.dataframe(df.head())
    st.write(f"### Kích thước dữ liệu: {df.shape}")
    st.write(f"### Số lượng giá trị bị thiếu: {df.isnull().sum().sum()}")
    
    st.write("### Thống kê tổng quan")
    st.write(df.describe())
    
    st.write("### Các cột đã được mã hóa")
    st.write(df[['Sex', 'Embarked']].head())
    
    st.write("### Tệp dữ liệu")
    st.write("- **Dữ liệu đã xử lý:**", data_path)
