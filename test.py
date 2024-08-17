from src.pipeline.predict_pipeline import PredictPipeline, CustomData

if __name__ == "__main__":
    pipeline = PredictPipeline()
    sample_input = CustomData("Sample text for prediction.")
    data = sample_input.get_data_as_data_frame()
    prediction = pipeline.predict(data)
    print(f"Prediction: {prediction}")
