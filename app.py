from dotenv import load_dotenv
load_dotenv()  # must be first before anything else

from ThermalSolarAnamolyDetection.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()