import pandas as pd
import random

random.seed(42)

low = [
    "The document contains standard operational language and routine administrative requirements.",
    "The supplier shall provide routine status updates during the service period.",
    "Employees may access the internal portal for ordinary business activities.",
    "The agreement describes standard reporting procedures and regular service reviews.",
    "Routine invoices shall be submitted according to the company's normal billing process.",
    "The customer may request ordinary support during the subscription period.",
    "The department shall maintain standard records for routine business operations.",
    "The service provider shall follow normal scheduling and communication procedures."
]

medium = [
    "The supplier must maintain appropriate records and provide compliance reports upon request.",
    "The agreement requires periodic review of access permissions and business records.",
    "The company may suspend access if required documentation is incomplete or outdated.",
    "The service provider must notify the customer of material operational changes.",
    "The contract requires additional approval before sensitive business information is shared.",
    "Records must be retained for the required period and made available for audit.",
    "The organization must review third-party access and document exceptions to policy.",
    "The supplier shall maintain security procedures and report significant incidents promptly."
]

high = [
    "The supplier may access confidential customer information without prior approval.",
    "The agreement permits transfer of sensitive personal data to an external third party.",
    "The company may disclose regulated information without obtaining the required authorization.",
    "The contract allows unrestricted access to financial records and confidential business information.",
    "The provider may retain sensitive customer data indefinitely without a defined deletion period.",
    "The organization may process personal information without documented consent or legal authorization.",
    "The agreement permits security controls to be bypassed when operationally convenient.",
    "The supplier is not required to report serious security incidents within a defined timeframe."
]

areas = [
    "data privacy",
    "access control",
    "financial compliance",
    "security",
    "records management",
    "vendor management"
]

rows = []
document_id = 1

for category, templates in [
    ("low", low),
    ("medium", medium),
    ("high", high)
]:

    for _ in range(200):

        area = random.choice(areas)

        if category == "low":
            ending = "No unusual exposure is identified."

        elif category == "medium":
            ending = "Additional compliance review may be appropriate."

        else:
            ending = "Manual compliance review is recommended before approval."

        text = (
            f"{random.choice(templates)} "
            f"This provision concerns {area}. "
            f"{ending}"
        )

        rows.append({
            "document_id": document_id,
            "text": text,
            "risk_category": category,
            "policy_area": area
        })

        document_id += 1

random.shuffle(rows)

df = pd.DataFrame(rows)

df.to_csv(
    "document_risk_dataset.csv",
    index=False
)

print("Dataset created successfully.")
print("Rows:", len(df))
print()
print(df["risk_category"].value_counts())
print()
print(df.head())