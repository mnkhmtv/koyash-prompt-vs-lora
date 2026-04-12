import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import streamlit as st

from src.inference.baseline import BaselineKoyashLLM
from src.inference.finetuned import FinetunedKoyashLLM
from src.eval.metrics import compute_perplexity, compute_rouge_l

def build_model_list(temperature: float) -> list:
    return [
        BaselineKoyashLLM(temperature=temperature),
        FinetunedKoyashLLM(temperature=temperature),
    ]


def load_default_system_prompt() -> str:
    path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'preprocessed', 'dataset.jsonl'
    )
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            first = f.readline().strip()
            if first:
                return json.loads(first).get("system_prompt", "Ты Koyash Ассистент")
    return "Ты Koyash Ассистент"

st.set_page_config(page_title="Koyash AI", page_icon="🌸", layout="wide")
st.title("🌸 Koyash AI — косметический консультант")

model_names = [m.name for m in build_model_list(0.3)]

with st.sidebar:
    st.header("⚙️ Настройки")

    selected_name = st.radio("Модель", model_names)

    st.divider()

    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.05)

models = build_model_list(temperature=temperature)
selected_model = next(m for m in models if m.name == selected_name)

system_prompt = st.text_area("System prompt:", value=load_default_system_prompt(), height=200)
user_input = st.text_area("Введите запрос:", height=150, placeholder="Опишите свой тип кожи, проблемы и бюджет...")
reference_input = st.text_area("Эталонный ответ для ROUGE-L (опционально):", height=80, placeholder="Вставьте экспертный ответ для сравнения...")

if st.button("✨ Получить рекомендацию", type="primary"):
    if not user_input.strip():
        st.warning("Введите запрос перед отправкой.")
    else:
        with st.spinner(f"Думаю ({selected_model.name})..."):
            try:
                response = selected_model.get_response(user_input, system_prompt=system_prompt)
                answer = selected_model.get_answer(response)
                st.markdown("### 💬 Ответ")
                st.markdown(answer)

                st.markdown("### 📊 Метрики")
                col1, col2 = st.columns(2)

                logprobs = response.logprobs or []
                ppl = compute_perplexity(logprobs)
                col1.metric("Perplexity ↓", f"{ppl:.2f}" if ppl == ppl else "N/A",
                            help="Чем ниже — тем увереннее модель в своём ответе")

                if reference_input.strip():
                    rouge = compute_rouge_l(answer, reference_input.strip())
                    col2.metric("ROUGE-L ↑", f"{rouge:.3f}",
                                help="Сходство с эталонным ответом (0–1)")
                else:
                    col2.metric("ROUGE-L", "—", help="Введите эталонный ответ выше")

            except Exception as e:
                st.error(f"Ошибка: {e}\n\nПроверьте, что Ollama запущен (`ollama serve`) и модель скачана.")

