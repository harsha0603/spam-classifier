test_data = [
    "Congratulations! You've won a $1000 gift card. Click here to claim your prize.",
    "Reminder: Your appointment is scheduled for tomorrow at 10 AM.",
    "Earn money from home! Sign up now to start earning extra income.",
    "Meeting agenda: Discuss project milestones and next steps.",
    "Exclusive offer just for you: Buy one get one free on all items.",
    "Important: Please review the attached document before the meeting.",
    "Don't miss out on this limited-time offer. Act fast to save big!",
    "Can you send me the report by end of day? Thanks.",
    "Get out of debt now! Call this number to learn more about our debt relief program.",
    "The team meeting is postponed to next week. Please update your calendar."
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