"""
Пример использования скрипта оценки модели.
"""

import os
from evaluate_dataset import evaluate_model

def main():
    # Пути (измените на свои)
    MODEL_PATH = r"C:\Users\USER\Desktop\OCR_MODELS\exp_4_model_64\best_acc_weights.pth"
    CHARSET_PATH = "configs\\charset.txt"
    
    # Пример для тестового датасета
    CSV_PATH = r"C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\labels.csv"
    ROOT_PATH = r"C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\img"
    
    # Размеры входных изображений (настройте под вашу модель)
    # Для модели exp_4_model_32:
    IMG_HEIGHT = 64   # Высота изображения
    IMG_WIDTH = 256   # Ширина изображения
    
    # Для других моделей раскомментируйте нужные размеры:
    # IMG_HEIGHT, IMG_WIDTH = 64, 256   # Для больших моделей
    # IMG_HEIGHT, IMG_WIDTH = 32, 256   # Для широких моделей
    
    print("🔥 Пример оценки модели на датасете")
    print(f"📐 Размеры изображений: {IMG_HEIGHT}x{IMG_WIDTH}")
    
    # Проверяем файлы
    for path, name in [(MODEL_PATH, "модель"), (CHARSET_PATH, "charset"), (CSV_PATH, "CSV"), (ROOT_PATH, "папка изображений")]:
        if not os.path.exists(path):
            print(f"❌ {name} не найден: {path}")
            return
    
    # Запускаем оценку
    try:
        evaluate_model(
            model_path=MODEL_PATH,
            charset_path=CHARSET_PATH,
            csv_path=CSV_PATH,
            root_path=ROOT_PATH,
            batch_size=16,
            img_h=IMG_HEIGHT,  # Высота входного изображения
            img_w=IMG_WIDTH    # Ширина входного изображения
        )
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()