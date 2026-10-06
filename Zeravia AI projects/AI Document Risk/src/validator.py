import numpy as np


def validate_prediction(
    prediction,
    confidence,
    threshold=0.60
):

    warnings = []

    if confidence < threshold:

        warnings.append(
            "Low confidence prediction"
        )

    if prediction is None:

        warnings.append(
            "No category predicted"
        )

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "manual_review": confidence < threshold,
        "warnings": warnings
    }


def get_top_terms(
    vectorizer,
    model,
    text,
    top_n=10
):

    X = vectorizer.transform(
        [text]
    )

    prediction = model.predict(
        X
    )[0]

    class_index = list(
        model.classes_
    ).index(prediction)

    coefficients = model.coef_[class_index]

    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    values = X.toarray()[0]

    scores = values * coefficients

    indices = np.argsort(
        scores
    )[::-1]

    terms = []

    for index in indices:

        if scores[index] <= 0:
            continue

        terms.append({
            "term": feature_names[index],
            "importance": float(
                scores[index]
            )
        })

        if len(terms) >= top_n:
            break

    return terms