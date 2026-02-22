import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st

from src.inference.baseline import BaselineKoyashLLM
from src.inference.finetuned import FinetunedKoyashLLM
from src.eval.metrics import compute_perplexity, compute_rouge_l

def build_model_list(system_prompt: str, temperature: float) -> list:
    return [
        BaselineKoyashLLM(system_prompt=system_prompt, temperature=temperature),
        FinetunedKoyashLLM(system_prompt=system_prompt, temperature=temperature),
    ]


def load_system_prompt() -> str:
    prompt_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'preprocessed', 'system_prompt.txt'
    )
    if os.path.exists(prompt_path):
        with open(prompt_path, encoding='utf-8') as f:
            return f.read()
    return "Ты Koyash Ассистент"

st.set_page_config(page_title="Koyash AI", page_icon="🌸", layout="wide")
st.title("🌸 Koyash AI — косметический консультант")

system_prompt = load_system_prompt()
model_names = [m.name for m in build_model_list(system_prompt, 0.3)]

with st.sidebar:
    st.header("⚙️ Настройки")

    selected_name = st.radio("Модель", model_names)

    st.divider()

    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.05)

models = build_model_list(system_prompt=system_prompt, temperature=temperature)
selected_model = next(m for m in models if m.name == selected_name)

user_input = st.text_area("Введите запрос:", height=150, placeholder="Опишите свой тип кожи, проблемы и бюджет...")
reference_input = st.text_area("Эталонный ответ для ROUGE-L (опционально):", height=80, placeholder="Вставьте экспертный ответ для сравнения...")

if st.button("✨ Получить рекомендацию", type="primary"):
    if not user_input.strip():
        st.warning("Введите запрос перед отправкой.")
    else:
        with st.spinner(f"Думаю ({selected_model.name})..."):
            try:
                response = selected_model.get_response(user_input)
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

