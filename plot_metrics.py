import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Настройки стиля для красивых графиков
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.autolayout": True
})

def shorten_model_name(name):
    """Укорачивает длинные названия моделей для легенды."""
    if ":" in name:
        return name.split(":")[0].replace("KoyashLLM", "")
    return name

def main():
    csv_file = "metrics_results.csv"
    
    if not os.path.exists(csv_file):
        print(f"Файл {csv_file} не найден!")
        return

    # Загружаем данные
    df = pd.read_csv(csv_file)
    if df.empty:
        print("CSV файл пуст.")
        return
        
    required_cols = {"id", "model_name", "perplexity", "rouge_l", "bert_score"}
    if not required_cols.issubset(df.columns):
        print(f"В файле не хватает колонок! Ожидаемые: {required_cols}")
        print(f"Текущие: {list(df.columns)}")
        return

    # Очищаем данные от возможных пустых или NaN значений в метриках
    df = df.dropna(subset=["perplexity", "rouge_l", "bert_score"])
    
    # Укорачиваем имена моделей
    df["model_short"] = df["model_name"].apply(shorten_model_name)
    
    models = df["model_short"].unique()
    print(f"Найдено моделей: {models}")
    
    metrics = {
        "perplexity": {"title": "Perplexity (Lower is better)", "minimize": True},
        "rouge_l": {"title": "ROUGE-L (Higher is better)", "minimize": False},
        "bert_score": {"title": "BERTScore F1 (Higher is better)", "minimize": False}
    }

    # Создаем папку для графиков
    os.makedirs("figures", exist_ok=True)

    # 1. Сводные метрики (Boxplot - распределение)
    print("Генерация boxplots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (metric, info) in zip(axes, metrics.items()):
        sns.boxplot(data=df, x="model_short", y=metric, ax=ax, width=0.5, palette="Set2")
        sns.stripplot(data=df, x="model_short", y=metric, ax=ax, color="black", alpha=0.5, size=5)
        
        ax.set_title(info["title"], pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Value")
    
    plt.suptitle("Metrics Distribution across Test Samples", y=1.05)
    plt.savefig("figures/metrics_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Средние значения метрик (Bar chart)
    print("Генерация средних значений...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    mean_df = df.groupby("model_short")[list(metrics.keys())].mean().reset_index()
    
    for ax, (metric, info) in zip(axes, metrics.items()):
        sns.barplot(data=mean_df, x="model_short", y=metric, ax=ax, palette="Set2")
        
        # Увеличиваем верхнюю границу графика, чтобы цифры не обрезались
        max_val = mean_df[metric].max()
        ax.set_ylim(0, max_val * 1.15)
        
        # Добавляем значения над столбцами
        for p in ax.patches:
            val = p.get_height()
            ax.annotate(f"{val:.3f}", (p.get_x() + p.get_width() / 2., val),
                        ha="center", va="bottom", fontsize=11, xytext=(0, 5),
                        textcoords="offset points")
            
        ax.set_title(f"Average: {info['title']}", pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("Average Value")
        
    plt.suptitle("Absolute Comparison of Average Metrics", y=1.05)
    plt.savefig("figures/metrics_averages.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Детальные результаты по каждому семплу (только если ID не очень много)
    unique_ids = df["id"].unique()
    if len(unique_ids) <= 20:
        print("Генерация графиков по каждому семплу...")
        fig, axes = plt.subplots(len(metrics), 1, figsize=(18, 4 * len(metrics)))
        
        for ax, (metric, info) in zip(axes, metrics.items()):
            sns.barplot(data=df, x="id", y=metric, hue="model_short", ax=ax, palette="Set2")
            ax.set_title(f"{info['title']} per Sample")
            ax.set_xlabel("Test Sample ID")
            ax.set_ylabel(metric)
            ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
            
        plt.tight_layout()
        plt.savefig("figures/metrics_per_sample.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("\nГрафики успешно сохранены в папку 'figures/':")
    print(" - figures/metrics_distribution.png (Распределения)")
    print(" - figures/metrics_averages.png (Средние значения)")
    if len(unique_ids) <= 20:
        print(" - figures/metrics_per_sample.png (По каждому семплу)")

if __name__ == "__main__":
    main()
