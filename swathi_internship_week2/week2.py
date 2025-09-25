# Student Performance Prediction using Logistic Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("STUDENT PERFORMANCE PREDICTION")
print("=" * 40)

# Step 1: Create simple dataset
print("\n1. Creating Dataset...")
data = {
    'Hours_Studied': [5, 2, 4, 1, 6, 3, 7, 2, 8, 1, 5, 3, 6, 4, 2, 7],
    'Attendance': [85, 60, 75, 50, 90, 70, 95, 55, 98, 45, 80, 65, 88, 78, 58, 92],
    'Pass_Fail': [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)
print("Dataset created!")
print(f"Total students: {len(df)}")
print("\nSample data:")
print(df.head())

# Step 2: Simple visualization
print("\n2. Basic Analysis...")
print(f"Students who passed: {df['Pass_Fail'].sum()}")
print(f"Students who failed: {len(df) - df['Pass_Fail'].sum()}")
print(f"Pass rate: {df['Pass_Fail'].mean():.1%}")

# Simple scatter plot
plt.figure(figsize=(8, 6))
passed = df[df['Pass_Fail'] == 1]
failed = df[df['Pass_Fail'] == 0]
plt.scatter(passed['Hours_Studied'], passed['Attendance'], color='green', label='Passed', s=100)
plt.scatter(failed['Hours_Studied'], failed['Attendance'], color='red', label='Failed', s=100)
plt.xlabel('Hours Studied')
plt.ylabel('Attendance (%)')
plt.title('Student Performance: Pass vs Fail')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Step 3: Prepare data
print("\n3. Preparing Data...")
X = df[['Hours_Studied', 'Attendance']]  # Input features
y = df['Pass_Fail']  # Output (Pass=1, Fail=0)

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training: {len(X_train)} students, Testing: {len(X_test)} students")

# Step 5: Train model
print("\n4. Training Model...")
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model trained!")

# Step 6: Test model
print("\n5. Testing Model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.1%}")

# Show test results
print("\nTest Results:")
for i in range(len(X_test)):
    hours = X_test.iloc[i]['Hours_Studied']
    attendance = X_test.iloc[i]['Attendance']
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    result = "✓" if actual == predicted else "✗"
    print(f"{result} Hours: {hours}, Attendance: {attendance}% -> Actual: {'Pass' if actual else 'Fail'}, Predicted: {'Pass' if predicted else 'Fail'}")

# Step 7: Try with new students
print("\n6. Testing with New Students...")
new_students = pd.DataFrame({
    'Hours_Studied': [6, 2, 8, 1],
    'Attendance': [90, 55, 95, 40]
})

print("Predicting for new students:")
predictions = model.predict(new_students)
for i, (hours, attendance, pred) in enumerate(zip(new_students['Hours_Studied'], 
                                                  new_students['Attendance'], 
                                                  predictions)):
    result = "Pass" if pred == 1 else "Fail"
    print(f"Student: {hours} hours/week, {attendance}% attendance -> {result}")

# Simple prediction function
def predict_performance(hours, attendance):
    new_data = pd.DataFrame({'Hours_Studied': [hours], 'Attendance': [attendance]})
    prediction = model.predict(new_data)[0]
    return "Pass" if prediction == 1 else "Fail"

print(f"\nExample: Student with 5 hours and 80% attendance -> {predict_performance(5, 80)}")