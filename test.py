test_data = [
    "Congratulations! You have won a $1,000 gift card. Click the link to claim your prize!"

"Urgent: Your account has been compromised. Click here to secure it immediately."

"Dear Customer, Your bank account is suspended. Verify your account details here."

"Get rich quick! Earn $5,000 a week from home with this simple method."

"Limited-time offer! Buy one, get one free on all electronics. Click now!"

"You have a new message waiting. Open this link to read it now!"

"Your package is on hold. Provide your address to ensure delivery."

"Work from home and make $500/day with no experience required! Click here to start."

"Act now! You are eligible for a low-interest rate on your loan. Apply today!"

"You’ve been selected for an exclusive discount. Redeem your code now!",
]

from src.pipeline.predict_pipeline import PredictPipeline

# Initialize the prediction pipeline
predict_pipeline = PredictPipeline()

# Make predictions
predictions = predict_pipeline.predict(test_data)

# Output the predictions
for message, prediction in zip(test_data, predictions):
    print(f"Message: {message}")
    print(f"Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
    print("------")