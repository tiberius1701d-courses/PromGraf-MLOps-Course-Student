import logging
import time

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="News Classifier API",
    description="API for classifying news articles into categories using a Hugging Face model.",
    version="1.0.0",
)

# Metrics definition.
registry = CollectorRegistry()

api_requests_total = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status_code"],
    registry=registry,
)

api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["endpoint", "method", "status_code"],
    registry=registry,
)

predictions_by_category = Counter(
    "predictions_by_category",
    "Number of predictions by category",
    ["category"],
    registry=registry,
)

model_accuracy_score = Gauge(
    "model_accuracy_score",
    "Current accuracy of the News Classifier model",
    registry=registry,
)

# NaN until the first evaluation, because 0 would read as zero accuracy.
model_accuracy_score.set(float("nan"))

# Gauges for precision, recall and F1 per category, set by /evaluate.
model_precision_score = Gauge(
    "model_precision_score",
    "Precision of the model per category on evaluation data",
    ["category"],
    registry=registry,
)
model_recall_score = Gauge(
    "model_recall_score",
    "Recall of the model per category on evaluation data",
    ["category"],
    registry=registry,
)
model_f1_score = Gauge(
    "model_f1_score",
    "F1 score of the model per category on evaluation data",
    ["category"],
    registry=registry,
)

# Buckets cover headlines (~50-100 characters) up to full articles (1000+).
input_text_length_chars = Histogram(
    "input_text_length_chars",
    "Length of the input text in characters",
    buckets=[20, 50, 100, 200, 500, 1000, 2000, 5000],
    registry=registry,
)

# Confidence scores lie between 0 and 1.
prediction_confidence_score = Histogram(
    "prediction_confidence_score",
    "Confidence score of model prediction",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
    registry=registry,
)

# Load the Hugging Face model
# This will download the model weights the first time it's run
MODEL_REVISION = "a5d6e7f999fb0d55fdbf156d861d4f4b45ea6b8a"
try:
    # Pin the model version; the course's call took whatever was newest.
    classifier = pipeline(
        "text-classification",
        model="dima806/news-category-classifier-distilbert",
        revision=MODEL_REVISION,
    )
    logger.info(
        "Hugging Face model loaded successfully: dima806/news-category-classifier-distilbert"
    )
except Exception as e:
    logger.error(f"Error loading Hugging Face model: {e}")
    raise RuntimeError("Error: Failed to load ML model.") from e

# The model reads at most 512 tokens and its tokenizer sets no limit itself.
MODEL_MAX_TOKENS = 512

categories = list(classifier.model.config.id2label.values())

# Create every category's counter at 0, so increase() also counts the first prediction.
for category in categories:
    predictions_by_category.labels(category=category)


class ArticleInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    category: str
    score: float


class EvaluationItem(BaseModel):
    text: str
    true_label: str


class EvaluationOutput(BaseModel):
    accuracy: float
    evaluated_items: int


@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the News Classifier API. Use /predict to classify articles."
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(article: ArticleInput):
    """
    Classify a news article based on its text content.
    """
    start_time = time.time()  # Start of the timer for the request duration

    status_code = "200"

    try:
        if not article.text:
            logger.warning("Received empty text for prediction.")
            status_code = "400"
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        input_text_length_chars.observe(len(article.text))

        # Perform prediction
        # Cut over-long texts to the model's input size; the course's call failed.
        results = classifier(article.text, truncation=True, max_length=MODEL_MAX_TOKENS)

        if not results:
            logger.error(
                f"Classifier returned empty results for text: {article.text[:50]}..."
            )
            status_code = "500"
            raise HTTPException(
                status_code=500, detail="Model could not classify the text."
            )

        # Assuming the first result is the most confident one
        predicted_category = results[0]["label"]
        confidence_score = results[0]["score"]

        # Counter increase for the predicted category
        predictions_by_category.labels(category=predicted_category).inc()
        prediction_confidence_score.observe(confidence_score)

        logger.info(
            f"Classified text (first 50 chars): '{article.text[:50]}...' into category: '{predicted_category}' with score: {confidence_score:.4f}"
        )
        return PredictionOutput(category=predicted_category, score=confidence_score)

    except HTTPException as e:
        status_code = str(e.status_code)
        raise
    except Exception as e:
        logger.error(
            f"Error during prediction for text: {article.text[:50]}... Error: {e}"
        )
        status_code = "500"
        raise HTTPException(
            status_code=500, detail=f"Prediction failed due to an internal error: {e}"
        )
    finally:
        end_time = time.time()
        # Request duration
        duration = end_time - start_time
        api_request_duration_seconds.labels(
            endpoint="/predict", method="POST", status_code=status_code
        ).observe(duration)
        api_requests_total.labels(
            endpoint="/predict", method="POST", status_code=status_code
        ).inc()


# Plain def: FastAPI runs the blocking model calls in a worker thread, so /metrics
# stays reachable. The course's endpoints use async def.
@app.post("/evaluate", response_model=EvaluationOutput)
def evaluate(items: list[EvaluationItem]):
    """Score the model on labeled items and set the accuracy and category gauges."""
    start_time = time.time()
    status_code = "200"

    try:
        if not items:
            raise HTTPException(
                status_code=400, detail="No items provided for evaluation."
            )
        true_labels = []
        predicted_labels = []
        for item in items:
            # Texts longer than the model's input are cut, not refused.
            prediction = classifier(
                item.text, truncation=True, max_length=MODEL_MAX_TOKENS
            )[0]["label"]
            predicted_labels.append(prediction)
            true_labels.append(item.true_label.upper())

        accuracy = accuracy_score(true_labels, predicted_labels)
        model_accuracy_score.set(accuracy)

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=categories,
            average=None,
            # NaN shows "no data" for a category the sample cannot score; the course's 0
            # showed a false score.
            zero_division=float("nan"),
        )
        for i, category in enumerate(categories):
            model_precision_score.labels(category=category).set(precision[i])
            model_recall_score.labels(category=category).set(recall[i])
            model_f1_score.labels(category=category).set(f1[i])

        logger.info(
            f"Model evaluated. Accuracy: {accuracy:.4f} on {len(true_labels)} items."
        )

        return EvaluationOutput(accuracy=accuracy, evaluated_items=len(true_labels))

    except HTTPException as e:
        status_code = str(e.status_code)
        raise
    except Exception:
        status_code = "500"
        raise
    finally:
        duration = time.time() - start_time
        api_request_duration_seconds.labels(
            endpoint="/evaluate", method="POST", status_code=status_code
        ).observe(duration)
        api_requests_total.labels(
            endpoint="/evaluate", method="POST", status_code=status_code
        ).inc()


@app.get("/metrics")
async def metrics(request: Request):
    """
    Expose Prometheus metrics.
    """
    return Response(content=generate_latest(registry), media_type="text/plain")
