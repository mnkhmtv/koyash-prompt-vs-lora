import random
import pandas as pd
import json

SEED = 42
TEST_SIZE = 10

consultations = pd.read_csv("data/raw/consultations_seed.csv")
products = pd.read_csv("data/raw/product_catalog.csv")

# Словарь: product_id → краткое описание для промпта
product_dict = {}
for _, p in products.iterrows():
    product_dict[p["product_id"]] = (
        f"{p['name']} ({p['brand']}) — {p['price_rub']}₽\n"
        f"Активные: {p['main_actives']}\n"
        f"Назначение: {p['functional_category']}"
    )

SYSTEM_FIRST = """You are a professional cosmetic consultant.
You are given a client profile and a pre-selected list of suitable products.
Write a detailed personalized consultation explaining each product, how to use it, and why it suits this client.
All listed products must appear in your response.

Client profile:
"""

SYSTEM_SECOND = """

Products for this client:
"""

rows = []

for _, row in consultations.iterrows():
    # Парсим айдишники
    ids = [i.strip() for i in str(row["products_recommended"]).split(",")]
    
    # Собираем описания продуктов
    products_text = "\n\n".join([
        product_dict[i] for i in ids if i in product_dict
    ])
    
    # Профиль клиента
    client_profile = (
        f"Age: {row['age']}\n"
        f"Skin type: {row['skin_type']}\n"
        f"Concerns: {row['concerns']}\n"
        f"Budget: {row['budget']}\n"
        f"Allergies: {row['allergies']}\n"
        f"Values: {row['values']}\n"
        f"Experience: {row['experience']}"
    )
    
    system_prompt = SYSTEM_FIRST + client_profile + SYSTEM_SECOND + products_text
    
    rows.append({
        "system_prompt": system_prompt,
        "prompt": "Составь персональную косметологическую консультацию.",
        "ideal_response": row["full_reasoning"]
    })

rng = random.Random(SEED)
indices = list(range(len(rows)))
rng.shuffle(indices)
test_idx = set(indices[:TEST_SIZE])

train_rows = [r for i, r in enumerate(rows) if i not in test_idx]
test_rows  = [r for i, r in enumerate(rows) if i in test_idx]

def dump(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

dump("data/preprocessed/train.jsonl", train_rows)
dump("data/preprocessed/test.jsonl", test_rows)

print(f"Готово: train={len(train_rows)}, test={len(test_rows)}")
print(f"Пример длины промпта: ~{len(rows[0]['system_prompt'].split())} слов")