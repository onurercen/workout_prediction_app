import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV dosyasını oku
df = pd.read_csv("gym_members_exercise_tracking.csv")

# İlk 5 satırı göster
print(df.head())

# Veri hakkında genel bilgi
print(df.info())

# Özet istatistik
print(df.describe())

# Yaş dağılımı grafiği
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], kde=True)
plt.title("Üyelerin Yaş Dağılımı")
plt.show()

print(df.columns)

# cinsiyete göre üye dagilimi
plt.figure(figsize=(6,5))
sns.countplot(x='Gender', data=df, palette='Set2')
plt.title("Cinsiyete Göre Üye Dağılımı")
plt.show()

#egzersiz türüne göre yakılan kalori
plt.figure(figsize=(10,6))
sns.barplot(x='Workout_Type', y='Calories_Burned', data=df, estimator='mean', ci=None, palette='coolwarm')
plt.title("Egzersiz Türüne Göre Ortalama Kalori Yakımı")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8,6))
sns.scatterplot(x='Age', y='BMI', hue='Gender', data=df)
plt.title("Yaş - BMI İlişkisi (Cinsiyete Göre)")
plt.show()

#egzersiz süresi - kalori

plt.figure(figsize=(8,6))
sns.regplot(x='Session_Duration (hours)', y='Calories_Burned', data=df, scatter_kws={'alpha':0.6})
plt.title("Egzersiz Süresi ile Kalori Yakımı İlişkisi")
plt.show()


#deneyim svy - workout sıklıgı
plt.figure(figsize=(7,5))
sns.barplot(x='Experience_Level', y='Workout_Frequency (days/week)', data=df, ci=None, palette='viridis')
plt.title("Deneyim Seviyesine Göre Haftalık Egzersiz Sıklığı")
plt.show()


# 1️⃣ Eksik verileri kontrol et
print("Eksik değerler:\n", df.isnull().sum())

# 2️⃣ Aykırı değerleri incele (örnek: Calories_Burned)
plt.figure(figsize=(8,5))
sns.boxplot(x='Calories_Burned', data=df)
plt.title("Calories_Burned Boxplot")
plt.show()

# İstersen diğer sayısal sütunlar için de boxplot ekleyebilirsin
numerical_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
                  'Session_Duration (hours)', 'Calories_Burned', 'Fat_Percentage', 
                  'Water_Intake (liters)', 'Workout_Frequency (days/week)', 'BMI']

for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"{col} Boxplot")
    plt.show()

# 3️⃣ Kategorik verileri sayısala çevir
df['Gender_Code'] = df['Gender'].map({'Male':0, 'Female':1})

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Workout_Code'] = le.fit_transform(df['Workout_Type'])

# 4️⃣ Numerik verileri normalize et
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# İşlem tamamlandı
print("Aşama 1 – Veri Ön İşleme tamamlandı ✅")

