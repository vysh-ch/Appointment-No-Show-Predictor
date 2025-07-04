import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import random

# 1. Data Loading and Feature Engineering
class FeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def process_data(self, data):
        """Process raw appointment data and create features"""
        df = data.copy()
        
        # Convert dates
        df['appointment_date'] = pd.to_datetime(df['appointment_date'])
        df['booking_date'] = pd.to_datetime(df['booking_date'])
        
        # Temporal features
        df['days_until_appointment'] = (df['appointment_date'] - df['booking_date']).dt.days
        df['appointment_hour'] = df['appointment_date'].dt.hour
        df['is_weekend'] = df['appointment_date'].dt.weekday >= 5
        df['month'] = df['appointment_date'].dt.month
        
        # Encode categorical variables
        categorical_cols = ['appointment_type', 'patient_gender', 'weather_condition']
        for col in categorical_cols:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
        
        # Patient history features
        df['no_show_rate'] = df.groupby('patient_id')['no_show'].transform('mean')
        df['appointment_frequency'] = df.groupby('patient_id')['appointment_date'].transform('count')
        
        # Weather impact
        df['weather_risk'] = df['weather_condition'].map({
            0: 0.1,  # clear
            1: 0.3,  # cloudy
            2: 0.6,  # rain
            3: 0.8   # snow
        })
        
        # Scale numerical features
        numerical_cols = ['age', 'days_until_appointment', 'appointment_hour', 
                         'no_show_rate', 'appointment_frequency']
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df

# 2. Prediction Model
class NoShowPredictor:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
    def train(self, X, y):
        """Train the prediction model"""
        self.model.fit(X, y)
        
    def predict(self, X):
        """Predict no-show probability"""
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.model.predict(X)
        probas = self.predict(X)
        print(classification_report(y, predictions))
        print(f"ROC AUC Score: {roc_auc_score(y, probas):.3f}")

# 3. Intervention Strategies
class InterventionManager:
    def __init__(self):
        self.strategies = {
            'high_risk': [
                'Send SMS reminder 24h before',
                'Personal phone call',
                'Offer transportation assistance'
            ],
            'medium_risk': [
                'Send SMS reminder 48h before',
                'Email reminder with rescheduling option'
            ],
            'low_risk': [
                'Standard email reminder'
            ]
        }
    
    def get_interventions(self, probability):
        """Assign intervention based on risk level"""
        if probability > 0.7:
            return self.strategies['high_risk']
        elif probability > 0.3:
            return self.strategies['medium_risk']
        else:
            return self.strategies['low_risk']

# 4. Scheduling Optimizer
class ScheduleOptimizer:
    def __init__(self):
        self.risk_threshold = 0.5
        
    def optimize_schedule(self, appointments, probabilities):
        """Optimize schedule based on risk predictions"""
        high_risk = appointments[probabilities > self.risk_threshold]
        recommendations = []
        
        for idx, prob in enumerate(probabilities):
            if prob > self.risk_threshold:
                patient_id = appointments.iloc[idx]['patient_id']
                appt_time = appointments.iloc[idx]['appointment_date']
                # Suggest earlier slots for high-risk patients
                new_time = appt_time - timedelta(hours=2)
                recommendations.append({
                    'patient_id': patient_id,
                    'original_time': appt_time,
                    'suggested_time': new_time,
                    'risk_score': prob
                })
        
        return pd.DataFrame(recommendations)

# 5. Main Pipeline
class NoShowPredictionSystem:
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.predictor = NoShowPredictor()
        self.intervention_manager = InterventionManager()
        self.schedule_optimizer = ScheduleOptimizer()
        
    def process_appointments(self, data):
        """Main pipeline for processing appointments"""
        # Feature engineering
        processed_data = self.feature_engineer.process_data(data)
        
        # Prepare features
        feature_cols = ['age', 'days_until_appointment', 'appointment_hour', 
                       'is_weekend', 'month', 'appointment_type', 
                       'patient_gender', 'weather_risk', 
                       'no_show_rate', 'appointment_frequency']
        
        X = processed_data[feature_cols]
        y = processed_data['no_show'] if 'no_show' in processed_data.columns else None
        
        # Train model if training data provided
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.predictor.train(X_train, y_train)
            self.predictor.evaluate(X_test, y_test)
        
        # Predict no-show probabilities
        probabilities = self.predictor.predict(X)
        
        # Generate interventions
        interventions = [self.intervention_manager.get_interventions(prob) for prob in probabilities]
        
        # Optimize schedule
        schedule_recommendations = self.schedule_optimizer.optimize_schedule(processed_data, probabilities)
        
        # Combine results
        results = processed_data.copy()
        results['no_show_probability'] = probabilities
        results['interventions'] = interventions
        
        return results, schedule_recommendations

# Example usage
if __name__ == "__main__":
    # Sample data creation
    np.random.seed(42)
    n_samples = 1000
    sample_data = pd.DataFrame({
        'patient_id': range(n_samples),
        'appointment_date': [datetime.now() + timedelta(days=random.randint(1, 30)) for _ in range(n_samples)],
        'booking_date': [datetime.now() - timedelta(days=random.randint(1, 14)) for _ in range(n_samples)],
        'age': np.random.randint(18, 80, n_samples),
        'patient_gender': np.random.choice(['M', 'F'], n_samples),
        'appointment_type': np.random.choice(['checkup', 'follow-up', 'specialist'], n_samples),
        'weather_condition': np.random.choice(['clear', 'cloudy', 'rain', 'snow'], n_samples),
        'no_show': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })
    
    # Initialize and run system
    system = NoShowPredictionSystem()
    results, schedule_recommendations = system.process_appointments(sample_data)
    
    # Print results
    print("\nPrediction Results:")
    print(results[['patient_id', 'no_show_probability', 'interventions']].head())
    print("\nScheduling Recommendations:")
    print(schedule_recommendations)